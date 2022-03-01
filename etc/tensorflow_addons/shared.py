import os

os.environ["TF_KERAS"] = '1'
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import Input
from tensorflow.keras.models import Model
import pickle
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (
    LayerNormalization, Layer, Flatten, Convolution1D, Concatenate, Activation, Dense, Embedding, GRU, LSTM, Dropout, MaxPooling1D,
    Lambda, AveragePooling1D, Bidirectional)

#-----------------------------------------------------------------------------------------------------------------------
class GELU(Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, x):
        return tf.nn.gelu(x)

    def compute_output_shape(self, input_shape):  # none, seq_len
        return input_shape

    def get_config(self):
        return super(GELU, self).get_config()

class MatDense(Layer):

    def __init__(self, embedding_dim, name = 'mat_dense', **kwargs):
        super(MatDense, self).__init__(name = name, **kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.dense1 = Dense(self.embedding_dim)
        super(MatDense, self).build(input_shape)

    def call(self, x):
        return self.dense1(x[:, 0, :])

    def compute_output_shape(self, input_shape):  # none, seq_len
        return [input_shape[0], self.embedding_dim]

    def get_config(self):
        config = super(LM_embed, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim
        })
        return config
    
class LM_embed(Layer):

    def __init__(self, max_len, vocab_size, embedding_dim, name = 'lm_embed', **kwargs):
        super(LM_embed, self).__init__(name = name, **kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.word_embed = self.add_weight(shape=(self.vocab_size + 1, self.embedding_dim),
                                          initializer='uniform',
                                          trainable=True)
        self.pos_feature = self.add_weight(shape=(self.max_len, self.embedding_dim),
                                           initializer='uniform',
                                           trainable=True)
        super(LM_embed, self).build(input_shape)

    def call(self, x):
        x = tf.cast(x, dtype=tf.int32)
        res = tf.nn.embedding_lookup(self.word_embed, x) + self.pos_feature  # 768 48
        return res

    def compute_output_shape(self, input_shape):  # none, seq_len
        return [input_shape[0], input_shape[1], self.embedding_dim]

    def get_config(self):
        config = super(LM_embed, self).get_config()
        config.update({
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim
        })
        return config

class word_embed(Layer):

    def __init__(self, max_len, vocab_size, embedding_dim, **kwargs):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        super(word_embed, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.word_embed = self.add_weight(shape = (self.vocab_size + 1, self.embedding_dim),
                                        initializer='uniform',
                                        trainable=True)
        super(word_embed, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        #print("word_embed layer is called", flush = True)
        x = tf.cast(x, dtype = tf.int32)
        res = tf.nn.embedding_lookup(self.word_embed, x)
        return res

    def compute_output_shape(self, input_shape): # none, seq_len
        return [input_shape[0], input_shape[1], self.embedding_dim]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim
        })
        return config

class pos_embed(Layer):

    def __init__(self, max_len, vocab_size, embedding_dim, **kwargs):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        super(pos_embed, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pos_feature = self.add_weight(shape = (self.max_len, self.embedding_dim),
                                           initializer='uniform',
                                           trainable=True)
        super(pos_embed, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        #print("pos_embed layer is called", flush = True)
        res = self.pos_feature    # 768 48
        res = tf.broadcast_to(res, [tf.shape(x)[0], self.max_len, self.embedding_dim])

        return res

    def compute_output_shape(self, input_shape): # none, seq_len
        return [input_shape[0], input_shape[1], self.embedding_dim]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim
        })
        return config


class FF(Layer):

    def __init__(self, is_training = True, **kwargs):
        self.is_training = is_training
        super(FF, self).__init__(**kwargs)

    def build(self, input_shape):
        self._model_dim = self._inner_dim = input_shape[-1]
        self._trainable = True
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")
        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")
        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        self.gelu1 = GELU()
        super(FF, self).build(input_shape)

    def call(self, inputs):
        inner_out = self.gelu1(tf.matmul(inputs, self.weights_inner) + self.bais_inner)
        outputs = tf.matmul(inner_out, self.weights_out) + self.bais_out
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(FF, self).get_config()

        config.update({
            'is_training': self.is_training
        })
        return config

class TransformerM(Layer):

    def __init__(self, layer_num, num_heads, is_training = True, name = 'atten_res', **kwargs):
        super(TransformerM, self).__init__(name = name, **kwargs)
        self.num_heads = num_heads
        self.layer_num = layer_num
        self.is_training = is_training

    def build(self, input_shape):
        self.heads = []
        self.ff = []
        self.lnorm = []
        for i in range(self.layer_num):
            self.heads.append(tfa.layers.MultiHeadAttention(head_size = input_shape[-1], num_heads = self.num_heads))
        for i in range(self.layer_num):
            self.ff.append(FF())
        self.gelu1 = GELU()
        for i in range(self.layer_num):
            self.lnorm.append(LayerNormalization())
            self.lnorm.append(LayerNormalization())
        super(TransformerM, self).build(input_shape)

    def call(self, x):
        for i in range(self.layer_num):
            x += self.heads[i]([x, x])
            x = self.lnorm[i * 2](x, training = self.is_training)
            x += self.ff[i](x)
            x = self.lnorm[i * 2 + 1](x, training = self.is_training)

        return x

    def compute_output_shape(self, input_shape):  # none, seq_len
        return input_shape

    def get_config(self):
        config = super(TransformerM, self).get_config()

        config.update({
            'num_heads': self.num_heads,
            'layer_num': self.layer_num,
            'is_training': self.is_training
        })
        return config

class SLSTM(Layer):

    def __init__(self, split_ratio, feature_dim, **kwargs):
        super(SLSTM, self).__init__(**kwargs)
        self.split_ratio = split_ratio
        self.feature_dim = feature_dim

    def build(self, input_shape):
        self.lstm_seq_len = input_shape[1] // self.split_ratio
        self.lstm_l = [Bidirectional(LSTM(self.feature_dim // 2)) for i in range(self.lstm_seq_len)]
        super(SLSTM, self).build(input_shape)

    def call(self, x):
        res = []
        for i in range(self.lstm_seq_len):
            start = i * self.split_ratio
            end = start + self.split_ratio
            res.append(self.lstm_l[i](x[:, start: end]))
        return tf.stack(res, axis = 1)

    def compute_output_shape(self, input_shape):  # none, seq_len
        return [input_shape[0], self.self.lstm_seq_len, self.feature_dim]

    def get_config(self):
        config = super(SLSTM, self).get_config()
        config.update({
            'split_ration': self.split_ratio,
            'feature_dim': self.feature_dim
        })
        return config

class RConcat(Layer):

    def __init__(self, axis = -1, is_training = True, **kwargs):
        self.is_training = is_training
        self.axis = axis
        super(RConcat, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n = len(input_shape)
        self.gelu1 = GELU()
        self.ratio = [self.add_weight(shape = (),
                                     initializer = 'uniform',
                                     trainable = True)
                      for i in range(self.n)]
        self.concat1 = Concatenate(axis = self.axis)
        self.dense1 = None
        self.lnorm1 = LayerNormalization()
        if self.axis == -1:
            self.dense1 = Dense(sum([input_shape[i][-1] for i in range(self.n)]))
        else:
            self.dense1 = Dense(input_shape[0][-1])
        super(RConcat, self).build(input_shape)
        

    def call(self, x):
        x = self.concat1([self.ratio[i] * x[i] for i in range(self.n)])
        x += self.dense1(x)
        x = self.gelu1(x)
        x = self.lnorm1(x, training = self.is_training)
        return x

    def compute_output_shape(self, input_shape):  # none, seq_len
        return [input_shape[0][0], input_shape[0][1], sum([input_shape[i][-1] for i in range(self.n)])]

    def get_config(self):
        config = super(RConcat, self).get_config()
        config.update({
            'is_training': self.is_training,
            'axis': self.axis
        })
        return config

#-----------------------------------------------------------------------------------------------------------------------
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

class Model_save(Callback):
    def __init__(self, model, valid_dir, train_path=None):
        self.model = model
        self.valid_dir = valid_dir
        self.train_path = train_path

    def on_train_begin(self, logs={}):
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        self.val_losses.append(val_loss)
        save_model(self.model, self.valid_dir + '/[' + str(epoch) + ']' + "%.3f|%.3f" % (float(val_loss), float(val_acc)))

    def on_train_end(self, logs={}):
        # save_model(self.model, self.valid_dir + '/[' + str(epoch) + ']' + str(val_loss))
        pass

#-----------------------------------------------------------------------------------------------------------------------
def concat_list(xlist):
    res = []
    for i in xlist:
        res.extend(i)
    return res

def compute_param(x):
    sp = x.shape
    res = 1
    for i in sp:
        res *= i
    return res

def load_data(File):
    seqs = np.load(File, allow_pickle=True)
    seqs = seqs.tolist()
    return seqs
#-----------------------------------------------------------------------------------------------------------------------
