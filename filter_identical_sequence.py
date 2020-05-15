import utils
import os
from Bio import SeqIO
import argparse
from tqdm import tqdm
import numpy as np


def search_identical(fasta):
    """
    given a fasta, return ndarray of identical pairs (seq_len= Seq.shape[1])  
    """
    Seq = np.array([SeqVector.seq for SeqVector in fasta])
    outer_ls = []
    keep = []
    discard = []

    for i in tqdm(range(Seq.shape[0])):
        inner = []
        for j in range(i+1,Seq.shape[0]):
            if np.sum(Seq[i,:] == Seq[j,:]) == Seq.shape[1] :
                inner.append(j)
        discard += inner
        outer_ls.append(inner)
        if i not in discard:
            keep.append(i)
        
    return keep,discard,outer_ls

def main():
    parser = argparse.ArgumentParser('filter out identical sequence and write a clean fasta file')
    parser.add_argument('in_path',type=str,help='please give an absolute path of in put fasta file',default='/home/ZwZ/database/M/linsi/M1_M2_all.fasta')
    parser.add_argument('out_path',type=str,help='the absolute path of filtered fasta file',default='/home/ZwZ/database/M/linsi/filter.fasta')
    args = parser.parse_args()

    ALL = utils.read_fasta(args.in_path,True)

    keep,discard,_ = search_identical(ALL) # the most important func
    all=[ALL[i] for i in keep]

    SeqIO.write(all,args.out_path,'fasta')

if __name__ == '__main__':
    main()