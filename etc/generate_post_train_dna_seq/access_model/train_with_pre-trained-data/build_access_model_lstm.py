import os
os.environ["TF_KERAS"] = '1'
from tensorflow_addons.shared import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input


def build_model(
                seq_len = 768,
                feature_len = 192,
                vocab_size = 6,
                embedding_dim = 32,
		        is_training = True
		       ):
    policy = tf.keras.mixed_precision.experimental.Policy('float32')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    input_feature = Input(shape=(seq_len, feature_len))
    input_element = Input(shape=(seq_len))

    word_embed1 = word_embed(256, vocab_size, embedding_dim)
    pos_embed1 = pos_embed(seq_len, vocab_size, embedding_dim)

    dna_vec = RConcat(is_training = True)([ input_feature, word_embed1(input_element), pos_embed1(input_element) ]) # 768 * 256

    dna_vec = SLSTM(24, 64)(dna_vec)
    dna_vec = Dropout(0.2)(dna_vec, training = is_training)
    dna_vec = SLSTM(8, 32)(dna_vec)
    dna_vec = Dropout(0.2)(dna_vec, training = is_training)
    dna_vec = SLSTM(1, 16)(dna_vec)
    dna_vec = Flatten()(dna_vec)
    dna_vec = Dense(4, activation='relu')(dna_vec)
    dna_vec = Dense(1, activation = 'sigmoid')(dna_vec)

    output = dna_vec

    model = Model(inputs= [ input_feature, input_element ],
                  outputs=output)
    return model

if __name__ == '__main__':
    build_model().summary()