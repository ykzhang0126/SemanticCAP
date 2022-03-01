import os
import sys
os.environ["TF_KERAS"] = '1'
sys.path.append(os.environ['d'])
from tensorflow_addons.shared import *


def build_model(max_len=256,
                   vocab_size=6,
                   embedding_dim=32,
                   hidden_dim=192,
                   is_training = True
                   ):
    policy = tf.keras.mixed_precision.experimental.Policy('float32')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    # tf.keras.mixed_precision.set_global_policy('float32')
    input_pre = Input(shape=(max_len,))
    input_post = Input(shape=(max_len,))

    LM_embed_i = LM_embed(max_len, vocab_size, embedding_dim)
    x_pre = LM_embed_i(input_pre)
    x_post = LM_embed_i(input_post)

    conv1 = Convolution1D(2 * embedding_dim, 24, padding='same', activation='relu')
    conv2 = Convolution1D(3 * embedding_dim, 16, padding='same', activation='relu')
    max1 = MaxPooling1D(2, 2)
    dropout1 = Dropout(0.2)
    concat1 = Concatenate()
    lnorm1 = LayerNormalization()

    x_pre = conv1(x_pre)
    x_pre = dropout1(x_pre, training = is_training)
    x_pre = max1(x_pre)
    x_pre = conv2(x_pre)
    x_pre = dropout1(x_pre, training = is_training)
    x_pre = max1(x_pre)
    x_post = conv1(x_post)
    x_post = dropout1(x_post, training = is_training)
    x_post = max1(x_post)
    x_post = conv2(x_post)
    x_post = dropout1(x_post, training = is_training)
    x_post = max1(x_post)
    x = concat1([x_pre, x_post])  # 64 192
    x = lnorm1(x, training = is_training)
    x = Dense(hidden_dim, activation='relu')(x)  # 64 192

    x = MultiHeadAttention_a(4, is_training)(x)
    x = Dense(hidden_dim // 4, activation='relu')(x)

    x = Flatten()(x)

    x = Dense(hidden_dim * 4, activation='relu')(x)
    x = Dense(hidden_dim, activation='relu')(x)

    # ------------------------------------------------------------------------------------
    output = x

    model = Model(inputs=[input_pre, input_post],
                  outputs=output)
    return model

