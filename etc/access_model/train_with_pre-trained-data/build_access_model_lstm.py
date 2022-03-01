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

    word_embed1 = word_embed(256, vocab_size, embedding_dim)
    pos_embed1 = pos_embed(seq_len, vocab_size, embedding_dim)
    gelu1 = GELU()

    dna_vec = RConcat(is_training = is_training)([ input_feature, word_embed1(input_element), pos_embed1(input_element) ]) # 768 * 256
    dna_vec = TransformerM(4, 3, is_training = is_training)(dna_vec)
    
    dna_vec = Convolution1D(256, 4, padding = 'same')(dna_vec)
    dna_vec = gelu1(dna_vec)
    dna_vec = Dropout(0.2)(dna_vec, training = is_training)
    dna_vec = AveragePooling1D(2, 2)(dna_vec) # 384 256
    
    dna_vec = Convolution1D(256, 8, padding = 'same')(dna_vec)
    dna_vec = gelu1(dna_vec)
    dna_vec = Dropout(0.2)(dna_vec, training = is_training)
    dna_vec = AveragePooling1D(4, 4)(dna_vec) # 96 256
    
    dna_vec = LayerNormalization()(dna_vec, training = is_training)
    dna_vec = Bidirectional(LSTM(128))(dna_vec)
    
    dna_vec = Dense(16)(dna_vec)
    dna_vec = gelu1(dna_vec)
    dna_vec = Dropout(0.2)(dna_vec, training=is_training)
    
    dna_vec = Dense(1, activation = 'sigmoid')(dna_vec)
    output = dna_vec

    model = Model(inputs= [ input_feature, input_element ],
                  outputs=output)
    return model

if __name__ == '__main__':
    build_model().summary()