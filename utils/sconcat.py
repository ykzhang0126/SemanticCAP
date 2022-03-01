import torch
import torch.nn as nn
import config
from utils.transformer import *

class SConcat(nn.Module):
    def __init__(self, feature_num, seq_len):
        super().__init__()
        self.ratios = nn.ParameterList([nn.Parameter(torch.ones(seq_len, 1)) for _ in range(feature_num)])

    def forward(self, x):
        res = []
        for i, feature in enumerate(x):
            res.append(self.ratios[i] * feature)
        return torch.cat(res, dim = 2)


if __name__ == '__main__':
    pass