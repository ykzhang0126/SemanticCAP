import os
import sys
sys.path.append(os.environ['d'])
import build_feature_model
from tensorflow_addons.shared import *


root_path = os.environ['d']
seq_len = int(sys.argv[1])
max_len = int(sys.argv[2])
batch_size = int(sys.argv[3])
feature_len = 192
tmode = 'fine_tune'

def process(name, xb, xa, xf):
    samples_num = xb.shape[0]
    start = 0
    end = batch_size
    while start < samples_num:
        BatchDSXpre = xb[start:end]
        BatchDSXpost = xa[start:end]
        x = []
#       with tf.device('/gpu:0'):
        for i in range(seq_len):
            print(f'\rposition {i}', end = '', flush = True)
            x.append(feature_model.predict([BatchDSXpre[:, i], BatchDSXpost[:, i]]))
            # print(x[-1])
        xf[start: end] = np.stack(x, axis=1)
        start += batch_size
        end += batch_size
        if end > samples_num:
            end = samples_num
        print(f'[{name}] {end}/{samples_num}', flush = True)
        del x, BatchDSXpre, BatchDSXpost


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

OutputDir = root_path + '/data/' + tmode + '/output' + str(max_len) + '/'
if not os.path.exists(OutputDir + '/checkpoint/'):
    os.makedirs(OutputDir + '/checkpoint/')

feature_model = build_feature_model.build_model(is_training = False)
load_model(feature_model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/feature_best_model')


train_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_pre.npy', mode='r')
train_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_pre.npy', mode='r',
                     shape=(train_xb.shape[0] // seq_len // max_len, seq_len, max_len))
train_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_post.npy', mode='r')
train_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_post.npy', mode='r',
                     shape=(train_xa.shape[0] // seq_len // max_len, seq_len, max_len))
valid_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_pre.npy', mode='r')
valid_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_pre.npy', mode='r',
                     shape=(valid_xb.shape[0] // seq_len // max_len, seq_len, max_len))
valid_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_post.npy', mode='r')
valid_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_post.npy', mode='r',
                     shape=(valid_xa.shape[0] // seq_len // max_len, seq_len, max_len))
test_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/test_x_pre.npy', mode='r')
test_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/test_x_pre.npy', mode='r',
                     shape=(test_xb.shape[0] // seq_len // max_len, seq_len, max_len))
test_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/test_x_post.npy', mode='r')
test_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/test_x_post.npy', mode='r',
                     shape=(test_xa.shape[0] // seq_len // max_len, seq_len, max_len))
train_xf = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_xf.npy',
    dtype = 'float32', mode='w+', shape = (train_xb.shape[0], seq_len, feature_len))
valid_xf = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_xf.npy',
    dtype = 'float32', mode='w+', shape = (valid_xb.shape[0], seq_len, feature_len))
test_xf = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/test_xf.npy',
    dtype = 'float32', mode='w+', shape = (test_xb.shape[0], seq_len, feature_len))



process('test', test_xb, test_xa, test_xf)
process('valid', valid_xb, valid_xa, valid_xf)
process('train', train_xb, train_xa, train_xf)
