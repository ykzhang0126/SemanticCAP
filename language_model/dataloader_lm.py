from multiprocessing import Process
from queue import Queue
import random
import time
import torch

import config


def reverse_base(sequence):
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


class dataloader_lm():
    def __init__(self):
        with open('../data/hg19', 'r', encoding='utf-8') as fin:
            self.data = fin.read()
            self.data_len = len(self.data)
        self.tensor_cpu = Queue(config.data_queue_max)
        # self.pw = Process(target = self.write, args=())
        # self.pw.start()

    def __del__(self):
        pass
        # self.pw.terminate()

    def random_context(self):
        random_idx = 0
        random_str = '\n'

        while True:
            random_idx = int(random.random() * self.data_len)
            if random_idx < config.context_len or random_idx > self.data_len - 1 - config.context_len:
                continue
            random_str = self.data[random_idx - config.context_len: random_idx + config.context_len + 1]
            if '\n' in random_str:
                continue
            break

        random_upstream = random_str[0: config.context_len]
        random_downstream = random_str[config.context_len + 1: ]
        random_base = random_str[config.context_len]
        
        if random.random() < 0.5:
            random_upstream, random_downstream = ''.join(reversed(random_downstream)), ''.join(reversed(random_upstream))
        if random.random() < 0.5:
            random_upstream = reverse_base(random_upstream)
            random_downstream = reverse_base(random_downstream)
            random_base = reverse_base(random_base)
            
        return list(random_upstream), list(random_downstream), random_base
    def drop_1(self, str_list, ratio):
        for i in range(len(str_list)):
            if random.random() < ratio:
                str_list[i] = 'MASK'
        return str_list
    def drop_x(self, str_list, ratio, mode):
        if mode == 'up':
            for i in range(0, int(len(str_list) * ratio)):
                str_list[i] = 'MASK'
        else:
            for i in range(len(str_list) - int(len(str_list) * ratio), len(str_list)):
                str_list[i] = 'MASK'
        return str_list
    def drop_context(self, stream, mode):
        drop = random.random()
        if drop < config.drop_1_prob:
            return self.drop_1(stream, config.drop_1_ratio)
        elif drop < config.drop_1_prob + config.drop_x_prob:
            return self.drop_x(stream, config.drop_x_ratio, mode)
        return stream

    def encode(self, str_list):
        if isinstance(str_list, list):
            return [config.symbol[s] for s in str_list]
        else:
            return config.symbol[str_list] - 1

    #写数据进程执行的代码：
    def write(self):
        data_batch_in = []
        data_batch_out = []
        for _ in range(config.lm_batch):
            upstream, downstream, base = self.random_context()
            upstream = self.drop_context(upstream, 'up')
            downstream = self.drop_context(downstream, 'down')
            data_batch_in.append(self.encode(['CLS'] + upstream + ['LOST'] + downstream))
            data_batch_out.append(self.encode(base))
        self.tensor_cpu.put((torch.tensor(data_batch_in, device = 'cpu'),
                            torch.tensor(data_batch_out, device = 'cpu')))


    #读数据进程执行的代码：
    def read(self):
        if self.tensor_cpu.empty():
            self.write()
        return self.tensor_cpu.get()


if __name__ == '__main__':
    x = dataloader_lm()
    ix = 1
    while True:
        print(ix)
        ix += 1
        print(x.read()[0])

