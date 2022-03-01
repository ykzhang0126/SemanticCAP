import os
import sys
sys.path.append(os.environ['d'])
from tensorflow_addons.shared import *
from lm_model import build_lm_model_gelu
import build_feature_model
import numpy as np



root_path = os.environ['d']
seq_len = int(sys.argv[1])
max_len = int(sys.argv[2])
max_epoch = int(sys.argv[3])
batch_size = int(sys.argv[4])
feature_len = 192
tmode = 'pre_train'


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

OutputDir = root_path + '/data/' + tmode + '/output' + str(max_len) + '/'
if not os.path.exists(OutputDir + '/checkpoint/'):
    os.makedirs(OutputDir + '/checkpoint/')

feature_model = build_feature_model.build_model(is_training = False)
load_model(feature_model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/feature_best_model')
lm_model = build_lm_model_gelu.build_model(is_training = False)
load_model(lm_model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/pre_best_model')
submodel = Model(inputs = lm_model.input, outputs = lm_model.get_layer(index = 14).output)
# submodel = Model(inputs = lm_model.input, outputs = lm_model.outputs)


train_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_pre.npy', mode='r')
train_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_pre.npy', mode='r',
                     shape=(train_xb.shape[0] // max_len, max_len))
train_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_post.npy', mode='r')
train_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_post.npy', mode='r',
                     shape=(train_xa.shape[0] // max_len, max_len))
valid_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_pre.npy', mode='r')
valid_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_pre.npy', mode='r',
                     shape=(valid_xb.shape[0] // max_len, max_len))
valid_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_post.npy', mode='r')
valid_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_post.npy', mode='r',
                     shape=(valid_xa.shape[0] // max_len, max_len))


def process(tmodel, name, xb, xa):
    samples_num = xb.shape[0]
    start = 0
    end = batch_size
    while start < samples_num:
        BatchDSXpre = xb[start:end]
        BatchDSXpost = xa[start:end]
        x = tmodel.predict([BatchDSXpre, BatchDSXpost])
        print(x)
        print(sum([1 if i != 0 else 0 for i in x[0]]))
        start += batch_size
        end += batch_size
        if end > samples_num:
            end = samples_num
        del x, BatchDSXpre, BatchDSXpost


process(submodel, 'train', train_xb, train_xa)
process(submodel, 'valid', valid_xb, valid_xa)

