import sys
import torch
import torch.nn as nn
gpu0 = torch.device('cpu')
if torch.cuda.is_available():
    gpu0 = torch.device('cuda:0')
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size = 512):
        super().__init__(vocab_size, embed_size, padding_idx = 0)

class PositionalEmbedding(nn.Embedding):
    def __init__(self, position_size, embed_size = 512):
        super().__init__(position_size, embed_size, padding_idx = 0)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(5, embed_size, padding_idx = 0)

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, input_len, context_len, dropout=0.1):

        super().__init__()
        self.token = TokenEmbedding(vocab_size = vocab_size, embed_size = embed_size)
        self.position = PositionalEmbedding(position_size = input_len + 1, embed_size = embed_size)
        self.segment = SegmentEmbedding(embed_size=embed_size)
        self.drop = nn.Dropout(p=dropout)
        self.position_label = torch.tensor([i + 1 for i in range(input_len)]).to(gpu0)
        self.segment_label = torch.tensor([1] + [2] * context_len + [3] + [4] * context_len).to(gpu0)

    def forward(self, sequence):
        x = self.token(sequence) + self.position(self.position_label) + self.segment(self.segment_label)
        return self.drop(x)

if __name__ == '__main__':
    pass