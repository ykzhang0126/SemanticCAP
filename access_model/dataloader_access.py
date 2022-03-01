from queue import Queue
import random
import torch
from torch.utils.data import Dataset, DataLoader
import config
from build_pred import *
gpu0 = torch.device('cpu')
'''
if torch.cuda.is_available():
    gpu0 = torch.device('cuda:0')
'''
class dataset_access(Dataset):

    def __init__(self, name, dataset_type, pred_model = None):
        self.name = name
        self.dataset_type = dataset_type
        self.training_data, self.validation_data, self.test_data = self.load_data()
        self.data = None
        if self.dataset_type == 'test':
            self.data = self.test_data
        elif self.dataset_type == 'valid':
            self.data = self.validation_data
        else:
            self.data = self.training_data
        self.pred_model = None
        if pred_model is not None:
            pred_model.eval()
            self.pred_model = pred_model.to(gpu0)
        self.pad0 = torch.zeros(1, config.lm_hidden_num, device = gpu0)

    def __getitem__(self, index):
        dna, target = self.data[index][0], self.data[index][1]
        dna_pad = self.pad(list(dna), config.dna_len)
        dna_pad2 = self.pad(list(dna), config.dna_len + 2 * config.context_len)

        if self.pred_model is None:
            return torch.tensor(self.encode(['CLS'] + dna_pad)), torch.tensor(target)
        return (torch.tensor(self.encode(['CLS'] + dna_pad), device = gpu0), self.pred(dna_pad2)), torch.tensor(target, device = gpu0)



    def __len__(self):
        return len(self.data)


    def pred(self, dna):
        pred_in = [self.pad0]
        with torch.no_grad():
            for i in range(config.context_len, config.context_len + config.dna_len):
                upstream = dna[i - config.context_len: i]
                downstream = dna[i + 1: i + config.context_len + 1]
                pred_in.append(self.pred_model(torch.tensor([self.encode(['CLS'] + upstream + ['LOST'] + downstream)], device = gpu0)))
        return torch.cat(pred_in, dim = 0)


    def load_data(self):
        seqs = []
        with open ('../data/' + self.name, 'r', encoding = 'utf-8') as fin:
            for line in fin:
                line = line.split()
                dna = line[0]
                dna_rev = ''.join(reversed(dna))
                access = int(line[1])
                seqs.append((dna, access))
                seqs.append((dna_rev, access))
                seqs.append((self.reverse_base(dna), access))
                seqs.append((self.reverse_base(dna_rev), access))

        samples_num = len(seqs)
        test_num = int(samples_num * config.test_ratio)
        valid_num = int(samples_num * config.valid_ratio)
        train_num = samples_num - test_num - valid_num
        return seqs[0: train_num], seqs[train_num: train_num + valid_num], seqs[train_num + valid_num: samples_num]

    def reverse_base(self, sequence):
        new_sequence = ''
        for s in sequence:
            if s == 'A':
                new_sequence += 'T'
            elif s == 'T':
                new_sequence += 'A'
            elif s == 'C':
                new_sequence += 'G'
            elif s == 'G':
                new_sequence += 'C'
        return new_sequence

    def encode(self, str_list):
        if isinstance(str_list, list):
            return [config.symbol[s] for s in str_list]
        else:
            return config.symbol[str_list] - 1

    def pad(self, dna, length):
        l = len(dna)
        if l == length:
            return dna
        if l > length:
            return dna[(l - length) // 2: (l - length) // 2 + length]
        lf = (length - l) // 2
        rf = length - l - lf
        return ['MASK'] * lf + dna + ['MASK'] * rf


def dataloader_access(name, mode, pred_model):
    return DataLoader(dataset_access(name, mode, pred_model),
                      batch_size = config.access_batch,
                      shuffle = True,
                      num_workers = 0,
                      pin_memory = True,
                      drop_last = True)

if __name__ == '__main__':
    x1 = dataloader_access('SNEDE0000EMX', 'train', None)

    for idx, i in enumerate(x1):
        print(i[1])
        print(idx)