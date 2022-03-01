
import tensorflow as tf
import os
import sys
import pickle
from lm_model import build_lm_model_gelu
import build_access_model_conv

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

seq_len = int(sys.argv[1])

def concat_list(xlist):
    res = []
    for i in xlist:
        res.extend(i)
    return res

def save_model(model, file_path):
    with open(file_path, 'wb') as fout:
        pickle.dump(model.get_weights(), fout)
    print('model saved to `%s`' % file_path)
    return 0

def load_model(model, file_path):
    if not os.path.exists(file_path):
        print('model weights file `%s` not found' % file_path)
        return -1
    with open(file_path, 'rb') as fin:
        model.set_weights(pickle.load(fin))
        print('model weights load from `%s`' % file_path)
    return 0

root_path = '/home/.ykzhang/data_process/'

lm_model = build_lm_model_gelu.build_model(is_training = False)
lm_model.summary()
if load_model(lm_model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/validBestModel') == -1:
    exit(-1)

model = build_access_model_conv.build_model(seq_len)
model.summary()

layer = [lm_model.get_layer(index = i) for i in [1, 2, 6, 8, 9, 10, 12, 13]]
weights = concat_list([i.get_weights() for i in layer])
layer_t = model.get_layer(index = 3)
assert [i.shape for i in weights] == [i.shape for i in layer_t.get_weights()]
layer_t.set_weights(weights)
save_model(model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/startpoint')
