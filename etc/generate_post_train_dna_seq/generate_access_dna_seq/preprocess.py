import sys
import os
import random

def check_dna(dna_seq):
    for i in dna_seq:
        if i != 'A' and i != 'G' and i != 'C' and i != 'T':
            return False
    return True

def pad_to_1280(dna):
    while len(dna) > 1280:
        dna = dna[1: len(dna)]
        if len(dna) > 1280:
            dna = dna[0: len(dna) - 1]

    dna = 'S' + dna[1: len(dna) - 1] + 'E'
    while len(dna) < 1280:
        dna = 'P' + dna
        if len(dna) < 1280:
            dna = dna + 'P'
    assert 'S' in dna
    assert 'E' in dna
    return dna


root_path = os.environ['d']
posfile = sys.argv[1]
negfile = sys.argv[2]

fp = open(posfile, 'r')
fn = open(negfile, 'r')
fo = open(root_path + '/data/fine_tune/SNEDE0000EMT', 'w')

data = []
for dna in fp:
    dna = str(dna.strip()).upper()
    if check_dna(dna) == False:
        continue
    if len(dna) < 50:
        continue
    dna = pad_to_1280(dna)
    data.append(dna + '\t' + '1\n')

for dna in fn:
    dna = str(dna.strip()).upper()
    if check_dna(dna) == False:
        continue
    if len(dna) < 50:
        continue
    dna = pad_to_1280(dna)
    data.append(dna + '\t' + '0\n')

random.shuffle(data)
fo.writelines(data)



fp.close()
fn.close()
fo.close()