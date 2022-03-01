import sys, os
# sys.path.append(os.environ['d'])
import torch
import torch.nn as nn
import config
from utils.transformer import *
from utils.mcnn import MultiCNN

class build_lm_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = BERTEmbedding(vocab_size = config.vocab_size, embed_size = config.embed_len,
                                   input_len = config.lm_input_len, context_len = config.context_len)
        self.mcnn = MultiCNN(config.lm_input_len, config.embed_len, config.lm_hidden_num, [1, 3, 5, 7, 9])
        '''
        self.bert = Performer(
            dim = config.lm_hidden_num,
            depth = config.lm_trans_num,
            heads = config.lm_heads_num,
            dim_head = config.lm_hidden_num * 2 // config.lm_heads_num,
            causal = False
        )
        '''
        self.bert = BERT(hidden = config.lm_hidden_num, attn_heads = config.lm_heads_num, n_layers = config.lm_trans_num)
        self.drop = nn.Dropout(config.lm_linear_dropout)

    def forward(self, x):
        # embedding the indexed sequence to sequence of vectors
        x = self.embed(x)
        x = self.mcnn(x)
        x = self.bert(x)
        x = self.drop(x[:, config.context_len + 1, :])
        return x

def build_pred():
    model = build_lm_feature()
    model_dict = model.state_dict()
    pre_model_dict = torch.load('../language_model/lm.pth')
    pre_model_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
    model_dict.update(pre_model_dict)
    model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    model = build_lm_feature()
    model_dict = model.state_dict()
    pre_model_dict = torch.load('../language_model/lm.pth')
    pre_model_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
    model_dict.update(pre_model_dict)
    model.load_state_dict(model_dict)
