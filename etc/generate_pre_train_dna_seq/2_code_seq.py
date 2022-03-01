#!/usr/bin/env python
#usage: ./this dna_file1 dna_file2...
import numpy as np
import sys
from Bio import SeqIO
import os

#np.random.seed(1)
#random.seed(1)

def process(fnames):
    seqs = {}

    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):  # parse(filename, format)
            dna_seq = str(record.seq).upper()
            if dna_seq.count('N') > 0:
                continue
            elif dna_seq.count('W') > 0:
                continue
            if dna_seq.upper() not in seqs:
                seqs[dna_seq] = []
            accession = record.description
            seqs[dna_seq].append(accession)
    seqs = np.array(list(seqs.keys()))
    return seqs


def featurize_seqs(seqs, vocabulary):
    size_vocabulary = len(vocabulary)
    start_int = size_vocabulary + 1 #special token
    end_int = size_vocabulary + 2

    #lens = np.array([ len(seq) for seq in seqs ])
    #order = np.argsort(lens)[::-1]
    #sorted_seqs = seqs[order]
    order = np.random.permutation(seqs.shape[0])
    sorted_seqs = seqs[order]   #shuffle seqs

    X = [
        [ start_int ] + [
            vocabulary[word] for word in seq
        ] + [ end_int ] for seq in sorted_seqs
    ]
    X = np.array(X)
    """
    print(X)
    sum0 = 0
    for i in range(len(X)):
        if 1 in X[i]:
            sum0+=1
    print(sum0)
    sys.exit()
    """
    return X

def setup(vocabulary):

    validSplit = 0.01
    ftrain = []
    for i in range(1, len(sys.argv)):
        ftrain.append(sys.argv[i])
	
    train_seqs = process(ftrain)    #get the seqs

    per = np.random.permutation(train_seqs.shape[0])
    train_seqs = train_seqs[per]    #shuffle

    samples = featurize_seqs(train_seqs, vocabulary)
    samples_num = samples.shape[0] 
    train_num = int(samples_num * (1-validSplit))
    print("train dna: %d samples, valid dna: %d samples" % (train_num, samples_num - train_num))
    #print(samples)
    return samples[: train_num], samples[train_num:]

if __name__ == '__main__':
    AAs = [
        'A', 'C','G','T'
    ]
    root_path = os.environ['d']

    vocabulary = { aa: idx + 1 for idx, aa in enumerate(AAs) }

    x_train, x_valid = setup(vocabulary)
    np.save(root_path + '/data/pre_train/dna_train_source', x_train)
    np.save(root_path + '/data/pre_train/dna_valid_source', x_valid)
