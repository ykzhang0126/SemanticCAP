#!/usr/bin/env python
#usage: ./this dna_file1 dna_file2...
import numpy as np
import sys
import os
import random

#np.random.seed(1)
#random.seed(1)


def process(fnames):
    seqs = []

    for fname in fnames:
        fp = open(fname, 'r')
        for record in fp:  # parse(filename, format)
            record = record.split()
            seqs.append([record[0], int(record[1])])
        fp.close()
    return seqs


def featurize_seqs(seqs, vocabulary):
    size_vocabulary = len(vocabulary)

    #lens = np.array([ len(seq) for seq in seqs ])
    #order = np.argsort(lens)[::-1]
    #sorted_seqs = seqs[order]


    X = [
            [ vocabulary[word] for word in seq[0] ] + 
            [ seq[1] ]
            for seq in seqs
    ]
    # X = np.array(X)
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

    validSplit = 0.1
    testSplit = 0.05
    ftrain = []
    for i in range(1, len(sys.argv)):
        ftrain.append(sys.argv[i])
	
    train_seqs = process(ftrain)    #get the seqs
    # print(len(train_seqs))
    random.shuffle(train_seqs)

    samples = featurize_seqs(train_seqs, vocabulary)
    samples_num = len(samples)
    train_num = int(samples_num * (1-validSplit - testSplit))
    valid_num = int(samples_num * validSplit)
    test_num = int(samples_num * testSplit)
    print("train dna: %d samples, valid dna: %d samples, test dna: %d samples" % (train_num, valid_num, test_num))
    #print(samples)
    return samples[: train_num], samples[train_num: train_num + valid_num], samples[train_num + valid_num: ]

if __name__ == '__main__':
    AAs = [
        'P', 'A', 'C','G','T', 'S', 'E'
    ]
    root_path = os.environ['d']

    vocabulary = { aa: idx for idx, aa in enumerate(AAs)}

    x_train, x_valid, x_test = setup(vocabulary)
    # print(x_train[0])
    # print(len(x_train[0]))
    np.save(root_path + '/data/fine_tune/dna_train_source', x_train)
    np.save(root_path + '/data/fine_tune/dna_valid_source', x_valid)
    np.save(root_path + '/data/fine_tune/dna_test_source', x_test)
