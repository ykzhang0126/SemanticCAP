import os
os.environ["TF_KERAS"] = '1'
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

    conv1 = Convolution1D(2 * embedding_dim, 4, padding='same')
    conv2 = Convolution1D(3 * embedding_dim, 8, padding='same')
    avg1 = AveragePooling1D(2, 2)
    # dropout1 = Dropout(0.2)
    dropout3 = Dropout(0.3)
    concat1 = RConcat(is_training = is_training)
    gelu1 = GELU()
    lnorm2 = LayerNormalization()
    lnorm3 = LayerNormalization()
    
    x_pre = conv1(x_pre)
    x_pre = gelu1(x_pre)
    x_pre = avg1(x_pre)
    x_pre = lnorm2(x_pre, training = is_training)
    x_pre = conv2(x_pre)
    x_pre = gelu1(x_pre)
    x_pre = avg1(x_pre)
    x_pre = lnorm3(x_pre, training = is_training)
    x_post = conv1(x_post)
    x_post = gelu1(x_post)
    x_post = avg1(x_post)
    x_post = lnorm2(x_post, training=is_training)
    x_post = conv2(x_post)
    x_post = gelu1(x_post)
    x_post = avg1(x_post)
    x_post = lnorm3(x_post, training = is_training)
    x = concat1([x_pre, x_post])  # 64 192

    x = TransformerM(layer_num = 4, num_heads = 2, is_training = is_training)(x)

    x = Flatten()(x)

    lnorm4 = LayerNormalization()
    x = Dense(hidden_dim * 4)(x)
    x = gelu1(x)
    x = Dense(hidden_dim)(x)
    x += Dense(hidden_dim)(x)
    x = gelu1(x)
    x = lnorm4(x, training = is_training)
    x = dropout3(x, training=is_training)
    # ------------------------------------------------------------------------------------
    out_dim = vocab_size - 2 + 1
    # embedding is over!
    x = Dense(out_dim, activation='softmax')(x)
    output = x

    model = Model(inputs=[input_pre, input_post],
                  outputs=output)
    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()
