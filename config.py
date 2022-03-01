#symbol
symbol = {
    'PAD': 0,
    'A': 1,
    'T': 2,
    'C': 3,
    'G': 4,
    'LOST': 5,
    'CLS': 6,
    'MASK': 7
}
vocab_size = len(symbol)

#pre-train
embed_len = 256
context_len = 512
lm_input_len = 2 * context_len + 2

drop_1_prob = 0.1
drop_1_ratio = 0.2
drop_x_prob = 0.15
drop_x_ratio = 0.4

lm_heads_num = 8
lm_trans_num = 8
lm_hidden_num = 256
lm_linear_dropout = 0.5

lm_batch = 4
data_queue_max = 4096
lm_step_samples = 1024
lm_show_samples = 1024
lm_save_samples = 102400   # 1000000
pre_lr = 1e-4
lm_model_out = './lm.pth'
lm_record_out = './lm.rec'

#access-train
dna_least_len = 50

test_ratio = 0.1
valid_ratio = 0.05

access_heads_num = 4
access_trans_num = 2
access_hidden_dim = lm_hidden_num
access_linear_dropout = 0.5
access_lr = 2e-5
#MT MU NO NP PC PH
access_data_name = 'SNEDE0000EPC'
access_model_out = './PC.pth'
dna_len = 767
access_epochs = 5

access_input_len = dna_len + 1
access_batch = 64