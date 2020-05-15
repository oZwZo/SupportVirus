from Bio import SeqIO,Seq,Align
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import re

def sites_to_keep(fasta):
    """discard identical sites across all samples
    """
    Seq = np.array([SeqVector.seq for SeqVector in fasta])

    # return a list of stable sites 
    size = len(Seq)
    same = []
    for i in range(Seq.shape[1]):
        if np.sum(Seq[:,i] == Seq[0,i]) == size:
            same.append(i)
    # the index of sites to keep
    return [i for i in range(Seq.shape[1]) if i not in same]

def sites2gene(sites,csv):
    for i in range(csv.shape[0]):
        if sites in range(csv.loc[i,'start'],csv.loc[i,'end']+1):
            return csv.loc[i,'gene']

def sites2genesite(sites,csv):
    for i in range(csv.shape[0]):
        if sites in range(csv.loc[i,'start'],csv.loc[i,'end']+1):
            return sites - csv.loc[i,'start']

def count_freq(array):
    """given a list or np array, return a DataFrame counting occurency of each element  
    """
    item = np.unique(array)
    freq = [np.sum(array == i) for i in item]
    return pd.DataFrame({'item':item,'frequency':freq})

def search_identical(fasta):
    """
    given a fasta, return ndarray of identical pairs 
    """
    Seq = np.array([SeqVector.seq for SeqVector in fasta])
    outer_ls = []
    keep = []
    discard = []

    for i in tqdm(range(Seq.shape[0])):
        inner = []
        for j in range(i+1,Seq.shape[0]):
            if np.sum(Seq[i,:] == Seq[j,:]) == Seq.shape[1]:
                inner.append(j)
        discard += inner
        outer_ls.append(inner)
        if i not in discard:
            keep.append(i)
        
    return keep,discard,outer_ls

def score_identity(fasta):
    N = len(fasta)
    identity = np.zeros((N,N),dtype=np.float32)

    for i in tqdm(range(N)):
        for j in range(N):
            identity[i,j]=aligner().score(fasta[i].seq,fasta[j].seq)/min(len(fasta[i].seq),len(fasta[j].seq))
    return identity

def most_n(array,n,max_or_min='max'):
    """
    return the most #n 'max/min' values and its index
    """
    pick=np.sort(array)[::-1][:n] if max_or_min=='max' else np.sort(array)[:n]
    return pick,[np.where(array == i)[0] for i in pick]

goose = ['snow goose', 'canada goose','barnacle goose','greylag goose', 
         'whitefronted goose','pink-footed goose','Goose','goose','Gs','gs','GS',
        'cackling goose','bean goose','Anser fabalis','White-fronted Goose']

list_of = lambda x,iterator : [itera.__getattribute__(x) for itera in iterator]   

def attribute2DataFrame(columns,iterator):
    return pd.DataFrame(dict(zip(columns,[list_of(x,iterator) for x in columns])))



########################################
#########       read fasta      ########
########################################

def read_fasta(path,formatting=False):
    fasta = list(SeqIO.parse(path,'fasta'))
        
    if formatting:
        for i in range(len(fasta)):
            format_argument(fasta[i])
    return fasta

def format_argument(SeqRecord):
    
    if '[host=' in SeqRecord.description:
        ori_description = SeqRecord.description
        host = re.match(r'.*\[host=(\w*)\]',SeqRecord.description).group(1)
        SeqRecord.description = SeqRecord.description.split('[host=')[0]
        format_argument(SeqRecord)
        SeqRecord.host = host
        SeqRecord.description = ori_description
        
    elif 'PLX' in SeqRecord.description:
        # sequence treatment
        SeqRecord.seq = Seq.Seq(SeqRecord.seq.__str__().replace('~','-'))
        
        # format for PLX sequence
        description = SeqRecord.description.split('_')
        SeqRecord.id = description[1]
        SeqRecord.host = description[2]
        SeqRecord.location = description[3]
        SeqRecord.name = '/'.join(description[2:6])
        SeqRecord.date = description[5]
        SeqRecord.subtype = description[6]
        SeqRecord.seq_len = len(SeqRecord.seq)
        SeqRecord.is_formated = True
    elif '|' in SeqRecord.description:
        # format of gisaid
        description = SeqRecord.description.split('|')
        info = description[0].strip().split('/')
          
        SeqRecord.host = info[1] 
        SeqRecord.location = info[2]
        SeqRecord.date = info[4]
           
        SeqRecord.name = description[0].strip()
        SeqRecord.gene = description[1].strip()
        SeqRecord.subtype = description[2].split('/')[1].strip()
        SeqRecord.id = description[4].strip() 
        SeqRecord.seq_len = len(SeqRecord.seq)
        SeqRecord.is_formated = True
    elif '_' in SeqRecord.description :
        description = SeqRecord.description.split('_')
        
        SeqRecord.id = description[0]
        SeqRecord.host = description[2]
        SeqRecord.name = '/'.join(description[1:-1]) if (description[-2] != 'Mixed') or (description[-2] != 'mixed') else '/'.join(description[1:-2])
        SeqRecord.date = description[-2] if (description[-2] != 'Mixed') or (description[-2] != 'mixed') else description[-3]
        SeqRecord.subtype = description[-1]
        
        SeqRecord.seq_len = len(SeqRecord.seq)
        SeqRecord.is_formated = True
    else:
        SeqRecord.is_formated = False

###########################################


#              -----------------------------
#  ===========|  convert  fasta to vector  |============
#             -----------------------------
def fasta_to_vec(fasta):
    
    """
    a final all in one function : 
     for a input parsed fasta list, convert all the SeqRecord object into SeqVector object
    
    input : list of SeqRecords  , which is list(SeqIO.parse(path of fasta),'fasta')
    
    output : list of SeqVectors , while all the property of virus are well inherited !
    """

    all_vec = []
    for i in range(len(fasta)):
        all_vec.append(SeqVector(fasta[i]))      # convert to SeqVector
        format_argument(all_vec[i])              # adding property
        
    return all_vec

class SeqVector(object):
    
    """
    a class that mimic SeqRecord but with tht seq property is replaced by sequence vector
    
    input : SeqRecord
    output : SeqVector
    
    """
    def __init__(self,SeqRecord):
        self.seq = SeqRecord.seq
        self.vector = aa_to_vec(SeqRecord.seq)   # convert to vector
        self.description = SeqRecord.description   
        # what we need is just the description property so that we can call the format_property
        
aa_vec_dict = {'A': 0, 'V': 1, 'I': 2, 'L': 3, 'M': 4, 'F': 5, 'Y': 6, 'W': 7, # non-polar aa
               'S': 8, 'T': 9, 'N': 10, 'Q': 11,                               # polar-non electrical charged aa
               'R': 12, 'H': 13, 'K': 14,'D': 15, 'E': 16,                     # positive and negative charged aa
               'C': 17, 'U': 18,'G': 19, 'P': 20,                              # special aa
               '-': 21,                                                        # gap
               'B':22,'J':22,'X':22,'Z':22}                                    # ambigirous               
def aa_to_vec(seq):
    '''
    A most important function of this notebook
    convert amino sequence to vector
    
    the dictionary are organised as followed
    '''
       
    n = len(seq)     # length -> columns of the matrix
    m = 23           # the total number alphabets included in sequence -> raws of the matrix
    
    seq_vector = np.zeros((m,n),dtype = np.uint64)
    
    for i in range(n):
        seq_vector[aa_vec_dict[seq[i]],i] = 1   #assign the row representing the aa with value 1 
    
    return seq_vector

# ====================================================


def confusion_matrix(y_true,y_predict,plot_out=False,class_label=['positive','negative']):
    
    '''
    compute the confusion matirx and return in the order of:
     
    |  True Positive , False Positive |
    |                                 |
    | False Negative , True Negative  |
    
    '''
    
    tn_count=sum(np.logical_and(y_true ==0,y_predict==0))   # true negative
    tp_count=sum(np.logical_and(y_true ==1,y_predict==1))   # true positive
    
    fn_count=sum(np.logical_and(y_true ==1,y_predict==0))   # false negative
    fp_count=sum(np.logical_and(y_true ==0,y_predict==1))   # false positive
    
    if plot_out == True:
        plt.figure(figsize=(10,10))
        plt.imshow(np.array([[tp_count,fp_count],[fn_count,tn_count]]),cmap=cm.Blues)
        plt.text(x=0,y=0,s=tp_count,fontsize=16)
        plt.text(x=1,y=0,s=fp_count,fontsize=16)
        plt.text(x=0,y=1,s=fn_count,fontsize=16)
        plt.text(x=1,y=1,s=tn_count,fontsize=16)
        plt.title('Confusion matrix',fontsize=20)
        plt.xticks([0,1],class_label,fontsize=14)
        plt.yticks([0,1],class_label,fontsize=14)
        plt.ylabel('Predict Value',fontsize=16)
        plt.xlabel('Actual Value',fontsize=16)
        plt.colorbar(fraction=0.046, pad=0.04)
    return np.array([[tp_count,fp_count],[fn_count,tn_count]])


tss = lambda x,y=False : torch.tensor(x,requires_grad=y)


def aligner():
    return Align.PairwiseAligner()

def list2dict(key_list,item_list):
    return dict(zip(key_list,item_list))