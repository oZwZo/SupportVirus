import numpy as np
import pandas as pd
from matplotlib import cm
from Bio import SeqIO
import os
import argparse
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

# sklearn package
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import joblib

from utils import *
from Model import *

parser = argparse.ArgumentParser()
parser.add_argument('--fasta_path',type=str,default='/home/ZwZ/database/M/linsi/M1_M2_560.fasta',
                    help='the path of ALIGNED fasta file containing sequences for binary classication')
parser.add_argument('--output_dir',type=str,default='Trained_model/',
                    help='the dir containing trained model')
parser.add_argument('--times_of_training',type=int,default=3,
                    help='the times of further training')
parser.add_argument('--combine_info',type=str,default='/home/ZwZ/database/M/linsi/M1_M2_all_trim_combination.csv',
                    help='the path of csv file noting how genes are concatenated')
parser.add_argument('--eps', '-e', type=float,default=0.03, help='elapse of ratio that can tolerent')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
parser.set_defaults(verbose=True)
args = parser.parse_args()

#         == Loading data == 

# directly to SeqVector
fasta = np.array(fasta_to_vec(read_fasta(args.fasta_path)))

#      == discard identical sites across all samples == 

keep2raw=sites_to_keep(fasta)

#            == define X and y (&ID) ==

# a func for binary label 
DKorGS=lambda host : 1 if host in goose else 0

X = np.array([SeqVector.vector[:,keep2raw].flatten() for SeqVector in fasta]) # flatten
ID = np.arange(len(fasta))                                      # ID for later use
y = np.array([DKorGS(SeqVector.host) for SeqVector in fasta])

# double check


#               == Splite Data == 
eps = args.eps

ratio_of = lambda y : np.sum(y==1)/len(y)
in_range = lambda x : (ratio_of(x)>true_ratio-eps)&(ratio_of(x)<true_ratio+eps)

true_ratio = ratio_of(y)
X_1,X_test,ID_1,ID_test = train_test_split(X,ID,test_size=0.1,shuffle=True)           # split test from other
X_train,X_val,ID_train,ID_val = train_test_split(X_1,ID_1,test_size=0.1,shuffle=True) # split train and val
y_train = y[ID_train];y_val = y[ID_val];y_test = y[ID_test]                           # from ID , yield y

while not in_range(y_train)&in_range(y_val)&in_range(y_test):
    X_1,X_test,ID_1,ID_test = train_test_split(X,ID,test_size=0.1,shuffle=True)           
    X_train,X_val,ID_train,ID_val = train_test_split(X_1,ID_1,test_size=0.1,shuffle=True) 
    y_train = y[ID_train];y_val = y[ID_val];y_test = y[ID_test]                           

print('\n Data Loading Succeed !','\n'*2,'Start spliting data','\n'*2)
print('------------------------------------------')
print('Ratio of positive sample in different sets')
print('------------------------------------------')
print(  'train |\t',ratio_of(y_train),
      '\n val  |\t',ratio_of(y_val),
      '\ntest  |\t',ratio_of(y_test))
print('------------------------------------------')

#      -------------\            /-----------
#      =============|   MODEL   |============
#      ------------/            \------------

def SV(**kwarg):
    return SVC_model(X_train,X_val,y_train,y_val,**kwarg)


# first training 
C_ls = [0.1,0.5,1,5,10,50,100,150,200,250,500,1000,2000,5000,10000]
SV_ls = [SV(**{'C':i,'kernel':'linear'}) for i in C_ls]
joblib.dump(SV_ls,os.path.join(args.output_dir,'primary_training.ModelList'))

pdf = PdfPages(os.path.join(args.output_dir,'primary_training.pdf'))
AUC_trace(SV_ls,'%s')
print('savefig...')
pdf.savefig()
pdf.close() 

if args.verbose:
    print('primary training')
    printting_training_result(SV_ls)

    
for repeat in tqdm(range(args.times_of_training)):
    new_C = tuning_C_given(SV_ls)
    SV_ls = [SV(**{'C':i,'kernel':'linear'}) for i in new_C]
    joblib.dump(SV_ls,os.path.join(args.output_dir,'the_{}.ModelList'.format(repeat+1)))    # save the whole list of SVM model
    pdf = PdfPages(os.path.join(args.output_dir,'the_{}th_training.pdf'.format(repeat+1)))
    AUC_trace(SV_ls,'%.3f')
    print('savefig...')
    pdf.savefig()
    pdf.close() 
    if args.verbose:
        print('\n the {}th training'.format(repeat+1))
        printting_training_result(SV_ls)
        
best_C = np.argmax([model.auc for model in SV_ls])
best_model = SV_ls[best_C]

# ===== write fasta =====
fasta = read_fasta(args.fasta_path)

DK_index=ID_train[best_model.support_][np.where(y[ID_train[best_model.support_]] == 0)[0]]
GS_index=ID_train[best_model.support_][np.where(y[ID_train[best_model.support_]] == 1)[0]]

DK_fasta = [fasta[i] for i in DK_index]
GS_fasta = [fasta[i] for i in GS_index]

SeqIO.write(DK_fasta,os.path.join(args.output_dir,'DK_support_virus.fasta'),'fasta')
SeqIO.write(GS_fasta,os.path.join(args.output_dir,'GS_support_virus.fasta'),'fasta')

# ====== omega of best model ==== 

omega = SVM_omega(best_model.coef_,threshold='scale',keep2raw=keep2raw)
DF=omega.DF

# ==== AUTO SEPERATE ================
comb_info = pd.read_csv(args.combine_info)
if comb_info is not None:
    DF.loc[:,'genes']=DF.raw_sites.apply(lambda x: sites2gene(x,comb_info))
    DF.loc[:,'genessites']=DF.raw_sites.apply(lambda x: sites2genesite(x,comb_info))
    
DF.to_csv(os.path.join(args.output_dir,'sites_info.csv'),index=False)

pdf = PdfPages(os.path.join(args.output_dir,'coef_info.pdf'))
omega.non_zero_hist()
pdf.savefig()
omega.letter_plot()
pdf.savefig()
pdf.close()  
