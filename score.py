from utils import aligner,read_fasta,identity
import numpy as np
import argparse
from numba import jit
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Adopt pairwise aligner to generate identity matrix')
    parser.add_argument('fasta_path',type=str,help='the path of fasta files to parse')
    parser.add_argument('out_path',type=str,help='output dir')
    args=parser.parse_args(['/home/ZwZ/database/M/linsi/M1_M2_unique.fasta','/home/ZwZ/database/M/linsi/M1_M2_unique.npy'])

    fasta = read_fasta(args.fasta_path)
    
    identity_M = identity(fasta)

    np.save(args.out_path,identity_M)

if __name__ == "__main__" :
    main()
