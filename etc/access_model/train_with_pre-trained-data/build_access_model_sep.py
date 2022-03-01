import os
os.environ["TF_KERAS"] = '1'
from tensorflow_addons.shared import *


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
    input_sep = tf.constant(0)
    
    word_embed1 = word_embed(256, vocab_size, embedding_dim)
    pos_embed1 = pos_embed(seq_len, vocab_size, embedding_dim)
    sep_embed1 = word_embed(256, 0, embedding_dim * 2 + feature_len)
    gelu1 = GELU()

    dna_vec = RConcat(is_training = is_training)([ input_feature, word_embed1(input_element), pos_embed1(input_element) ]) # 768 * 256
    sep_feature = tf.broadcast_to(sep_embed1(input_sep), [tf.shape(dna_vec)[0], 1, embedding_dim * 2 + feature_len])
    dna_vec = Concatenate(axis = 1)([dna_vec, sep_feature])
    
    dna_vec = TransformerM(4, 8, is_training = is_training)(dna_vec)
    
    dna_vec = MatDense(16)(dna_vec)
    
    dna_vec = Dense(1, activation = 'sigmoid')(dna_vec)
    output = dna_vec

    model = Model(inputs= [ input_feature, input_element ],
                  outputs=output)
    return model