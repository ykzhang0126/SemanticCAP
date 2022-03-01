import sys
import random
sys.path.append('../../')
import config

def check_dna(s):
    for ch in s:
        if ch != 'A' and ch != 'T' and ch != 'C' and ch != 'G':
            return False
    return True

def get_seq(filename, suffix):
    seqs = []
    fin = open(filename, 'r', encoding = 'utf-8')
    for line in fin:
        line = line.strip().upper()
        if len(line) >= config.dna_least_len and check_dna(line) == True:
            seqs.append(line + ' ' + suffix)
    fin.close()
    return seqs



def split(filename):
    pos_seqs = get_seq('../raw_data/' + filename + '_pos', suffix = '1')
    neg_seqs = get_seq('../raw_data/' + filename + '_neg', suffix = '0')
    samples = pos_seqs + neg_seqs
    random.shuffle(samples)

    with open('../' + filename, 'w', encoding = 'utf-8') as fout:
        for line in samples:
            fout.write(line)
            fout.write('\n')
    print(filename + ' is ok, ' + str(len(samples)) + ' samples written.')


split('SNEDE0000EMT')
split('SNEDE0000EMU')
split('SNEDE0000ENO')
split('SNEDE0000ENP')
split('SNEDE0000EPC')
split('SNEDE0000EPH')