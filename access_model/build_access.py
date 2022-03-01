import sys, os
# sys.path.append(os.environ['d'])
import torch
import torch.nn as nn
import config
from utils.transformer import *
from utils.mcnn import MultiCNN
from utils.sconcat import SConcat
from utils.seq_embedding import SEQEmbedding

class build_access_class(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = SEQEmbedding(vocab_size = config.vocab_size, embed_size = config.embed_len,
                                   input_len = config.access_input_len)
        self.mcnn = MultiCNN(config.access_input_len, config.embed_len, config.access_hidden_dim, [1, 3, 5, 7, 9])

        self.sc1 = SConcat(2, config.access_input_len)
        self.lin1 = nn.Linear(config.lm_hidden_num + config.access_hidden_dim, config.access_hidden_dim)
        self.bert = BERT(hidden = config.access_hidden_dim, attn_heads = config.access_heads_num, n_layers = config.access_trans_num)
        self.drop = nn.Dropout(config.access_linear_dropout)
        self.lin2 = nn.Linear(config.lm_hidden_num, 1)

    def forward(self, x):
        # embedding the indexed sequence to sequence of vectors
        # x, y = x[0], x[1]
        x = self.embed(x)
        x = self.mcnn(x)
        # x = self.sc1((x, y))
        # x = self.lin1(x)
        x = self.bert(x)
        x = self.drop(x[:, 0, :])
        x = self.lin2(x)
        return x

def build_access():
    model = build_access_class()
    model_dict = model.state_dict()
    pre_model_dict = torch.load('../language_model/lm.pth')
    model_dict['embed.token.weight'] = pre_model_dict['embed.token.weight']
    model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    model = build_access()
    print(model)
