import numpy as np
import pandas as pd
import os
import argparse
import joblib

from utils import *
from Model import *

# ===========================
parser = argparse.ArgumentParser('the python script to seperate sites info into gene specific')
parser.add_argument('--fasta_path',type=str,default='/home/ZwZ/database/M/linsi/M1_M2_560.fasta',
                    help='the path of ALIGNED fasta file containing sequences for binary classication')
parser.add_argument('--combine_info',type=str,default='/home/ZwZ/database/M/linsi/M1_M2_all_trim_combination.csv',
                    help='the path of csv file noting how genes are concatenated')
parser.add_argument('--model_dir',type=str,default='Trained_model/',
                    help='the dir containing trained model')
parser.add_argument('--model',type=str,default='the_3.ModelList',
                    help='the Model List ')
parser.add_argument('--which',type=int,default=None,help='the index of model in model list')
args = parser.parse_args()
# ===========================

#  == Loading data == 

# directly to SeqVector
fasta = np.array(fasta_to_vec(read_fasta(args.fasta_path)))

comb_info = pd.read_csv(args.combine_info)

print('----------------------\n')
print('\n *NOTED : comb_info is read from {} \n'.format(args.combine_info))
print('----------------------\n')
#  == discard identical sites across all samples == 

keep2raw=sites_to_keep(fasta)
    
# ====|>> loding model <<|====
    
model_path = os.path.join(args.model_dir,args.model)
SV_ls = joblib.load(model_path)
    
# the model to pick
if args.which is not None:
    idx = args.which
else:
    idx = np.argmax([model.auc for model in SV_ls])
    
omega=SVM_omega(SV_ls[idx].coef_,keep2raw=keep2raw)
    
# 
DF=omega.DF
    
DF.loc[:,'genes']=DF.raw_sites.apply(lambda x: sites2gene(x,comb_info))
DF.loc[:,'genessites']=DF.raw_sites.apply(lambda x: sites2genesite(x,comb_info))

DF_name = '{}_{}_seperated.csv'.format(args.model.split('.')[0],idx)
DF.to_csv(os.path.join(args.model_dir,DF_name))