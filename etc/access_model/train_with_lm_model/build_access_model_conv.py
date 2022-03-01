import os
import sys
os.environ["TF_KERAS"] = '1'
sys.path.append(os.environ['d'])
from tensorflow_addons.shared import *



class LM_layer(Layer):  # input: [None, seq_len, max_len] * 2 output: [None, seq_len, hidden_dim]
    def __init__(self, seq_len, max_len, hidden_dim, embedding_dim, vocab_size, **kwargs):
        self.seq_len = seq_len
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        super(LM_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.word_embed_i = word_embed(self.max_len, self.vocab_size, self.embedding_dim)
        self.pos_embed_i = pos_embed(self.max_len, self.vocab_size, self.embedding_dim)
        self.conv1 = Convolution1D(2 * self.embedding_dim, 24, padding='same', activation='relu')
        self.conv2 = Convolution1D(3 * self.embedding_dim, 16, padding='same', activation='relu')
        self.max1 = MaxPooling1D(2, 2)
        self.dropout1 = Dropout(0.2)
        self.concat1 = Concatenate()
        self.dense1 = Dense(self.hidden_dim, activation='relu')
        self.multiheadattention1 = MultiHeadAttention_a(4)
        self.dense2 = Dense(self.hidden_dim // 4, activation='relu')
        self.flatten1 = Flatten()
        self.dense3 = Dense(self.hidden_dim * 4, activation='relu')
        self.dense4 = Dense(self.hidden_dim, activation='relu')
        self.unstack1 = Lambda(lambda x: tf.unstack(x, axis=1))
        self.expand1 = Lambda(lambda x: tf.reshape(x, [-1, 1, self.hidden_dim]))
        self.concat2 = Concatenate(axis=1)
        super(LM_layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #print("lm layer is called", flush = True)
        input_pres = self.unstack1(x[0])
        input_posts = self.unstack1(x[1])
        input_ys = x[2]
        xx_list = []
        idx = 1
        for i in range(self.seq_len):
            input_pre = input_pres[i]
            input_post = input_posts[i]
            print("\rbuild model process: %.1f%%" % (idx / self.seq_len * 100), end = '', flush = True)
            idx += 1
            x_pre = self.word_embed_i(input_pre) + self.pos_embed_i(input_pre)
            x_post = self.word_embed_i(input_post) + self.pos_embed_i(input_post)
            x_pre = self.conv1(x_pre)
            x_pre = self.dropout1(x_pre)
            x_pre = self.max1(x_pre)
            x_pre = self.conv2(x_pre)
            x_pre = self.dropout1(x_pre)
            x_pre = self.max1(x_pre)
            x_post = self.conv1(x_post)
            x_post = self.dropout1(x_post)
            x_post = self.max1(x_post)
            x_post = self.conv2(x_post)
            x_post = self.dropout1(x_post)
            x_post = self.max1(x_post)
            xx = self.concat1([x_pre, x_post])  # 128 384
            xx = self.dense1(xx)  # 128 192
            xx = self.multiheadattention1(xx)
            xx = self.dense2(xx)
            xx = self.flatten1(xx)
            xx = self.dense3(xx)
            xx = self.dense4(xx)
            xx = self.expand1(xx)
            xx_list.append(xx)
        print("\nbuild model done", flush = True)
        return self.concat2(xx_list), self.word_embed_i(input_ys)

    def compute_output_shape(self, input_shape):
        batchs = input_shape[0][0]
        return [[batchs, self.seq_len, self.hidden_dim], [batchs, self.seq_len, self.embedding_dim]]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'seq_len': self.seq_len,
            'max_len': self.max_len,
            'hidden_dim': self.hidden_dim,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim
        })
        return config


def build_model(
                seq_len = 768,
                max_len = 256,
                vocab_size = 4 + 2,
		       embedding_dim=32,
		       hidden_dim=192,
               is_training = True
		       ):
    policy = tf.keras.mixed_precision.experimental.Policy('float32')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    # tf.keras.mixed_precision.set_policy(policy)
    # tf.keras.mixed_precision.set_global_policy('float32')
    input_pres = Input(shape=(seq_len, max_len))
    input_posts = Input(shape=(seq_len, max_len))
    input_ys = Input(shape = (seq_len, ))

    #print("*******************model being build******************", flush = True)
    # out 192 + 32-----------------------------------------------------------------------------
    x, y = LM_layer(seq_len, max_len, hidden_dim, embedding_dim, vocab_size)([input_pres, input_posts, input_ys])
    # out 32------------------------------------------------------------------------------
    pos_embed2 = Embedding(seq_len + 1, embedding_dim)
    pos2 = tf.constant([i for i in range(1, seq_len + 1)])
    z = pos_embed2(pos2)
    z = tf.broadcast_to(z, [tf.shape(x)[0], seq_len, embedding_dim])
    # print('******7777*********')
    # print(z.shape)

    # dna access model------------------------------------------------------------------------------
    dna_vec = Concatenate()([ x, y, z ]) # 768 * 256
    # print(dna_vec.shape)
    dna_vec = LayerNormalization()(dna_vec, training = is_training)    
    dna_vec = Dense(256, activation = 'relu')(dna_vec)

    dna_vec = Convolution1D(64, 24, padding = 'same', activation='relu')(dna_vec)
    dna_vec = Dropout(0.2)(dna_vec)
    dna_vec = MaxPooling1D(8, 8)(dna_vec) # 96 * 64
    dna_vec = Convolution1D(32, 12, padding = 'same', activation='relu')(dna_vec)
    dna_vec = Dropout(0.2)(dna_vec)
    dna_vec = MaxPooling1D(4, 4)(dna_vec) # 24 * 32
    dna_vec = Flatten()(dna_vec)
    dna_vec = Dense(128, activation = 'relu')(dna_vec)
    dna_vec = Dense(32, activation = 'relu')(dna_vec)
    dna_vec = Dense(4, activation = 'relu')(dna_vec)
    dna_vec = Dense(1, activation = 'sigmoid')(dna_vec)

    output = dna_vec

    model = Model(inputs=[ input_pres, input_posts, input_ys ],
                        outputs=output)
    return model
    
