import torch
import torch.nn as nn
import config
from utils.transformer import *

class MultiCNN(nn.Module):
    def __init__(self, seq_len, input_dim, output_dim, kernels):
        super().__init__()
        self.cnns = nn.ModuleList([nn.Conv1d(input_dim, output_dim, padding = (k - 1) // 2, kernel_size = k) for k in kernels])
        self.lns = nn.ModuleList([LayerNorm(output_dim) for _ in range(len(kernels))])
        self.ratios = nn.ParameterList([nn.Parameter(torch.ones(seq_len, output_dim)) for _ in range(len(kernels))])

    def forward(self, x):
        res = []
        for ln, ratio, cnn in zip(self.lns, self.ratios, self.cnns):
            res.append(ratio * ln(cnn(x.permute(0, 2, 1)).permute(0, 2, 1)))
        return sum(res)


if __name__ == '__main__':
    x = MultiCNN(2,2,3,[1])