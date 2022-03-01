import os
import sys
sys.path.append(os.environ['d'])
from tensorflow_addons.shared import *
from lm_model import build_lm_model_gelu
import build_feature_model
# import build_access_model_conv
# import build_access_model_lstm
import build_access_model_sep
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


root_path = '/home/ykzhang/data_process/'
# root_path = '../../'

def init_feature_model():
    lm_model = build_lm_model_gelu.build_model(is_training = False)
    lm_model.summary()
    feature_model = build_feature_model.build_model(is_training = False)
    feature_model.summary()
    if load_model(lm_model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/pre_best_model') == -1:
        print('lm model weights not exist!')
        exit(-1)
    llayer = [1, 2, 6, 7, 8, 9, 10, 12, 13, 14, 16]
    layer = [lm_model.get_layer(index = i) for i in llayer]
    weights = [i.get_weights() for i in layer]

    tlayers = [feature_model.get_layer(index = i) for i in llayer]
    tweights = [i.get_weights() for i in layer]

    params = 0
    for i, tlayer in enumerate(tlayers):
        tlayer.set_weights(weights[i])
        # print(weights[i])
        # print(feature_model.get_layer(index = llayer[i]).get_weights())
        # assert feature_model.get_layer(index = llayer[i]).get_weights() == lm_model.get_layer(index = llayer[i]).get_weights()
        layer_param = sum([compute_param(w) for w in weights[i]])
        params += layer_param
        print(f'{layer_param} parameters have been set')

    save_model(feature_model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/feature_best_model')
    print(f'{params} parameters have been set')

def init_conv_model():
    lm_model = build_lm_model_gelu.build_model(is_training=False)
    lm_model.summary()
    model = build_access_model_conv.build_model(is_training=False)
    model.summary()
    if load_model(lm_model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/pre_best_model') == -1:
        print('lm model weights not exist!')
        exit(-1)
    # print(lm_model.get_layer(index = 1).get_weights()[0].shape)
    # print(model.get_layer(index=2).get_weights()[0].shape)
    model.get_layer(index = 2).set_weights([lm_model.get_layer(index = 1).get_weights()[0]])
    save_model(model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/fine_best_model')
    layer_param = compute_param(lm_model.get_layer(index = 1).get_weights()[0])
    print(f'{layer_param} parameters have been set')
    '''
    layer = [lm_model.get_layer(index = i) for i in [1, 2, 6, 8, 9, 10, 12, 13]]
    weights = concat_list([i.get_weights() for i in layer])
    layer_t = model.get_layer(index = 3)
    assert [i.shape for i in weights] == [i.shape for i in layer_t.get_weights()]
    layer_t.set_weights(weights)
    save_model(model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/startpoint')
    '''

def init_lstm_model():
    lm_model = build_lm_model_gelu.build_model(is_training=False)
    lm_model.summary()
    model = build_access_model_lstm.build_model(is_training=False)
    model.summary()
    if load_model(lm_model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/pre_best_model') == -1:
        print('lm model weights not exist!')
        exit(-1)
    # print(lm_model.get_layer(index = 1).get_weights()[0].shape)
    # print(model.get_layer(index=2).get_weights()[0].shape)
    model.get_layer(index = 2).set_weights([lm_model.get_layer(index = 1).get_weights()[0]])
    save_model(model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/conv_best_model')
    layer_param = compute_param(lm_model.get_layer(index = 1).get_weights()[0])
    print(f'{layer_param} parameters have been set')
    '''
    layer = [lm_model.get_layer(index = i) for i in [1, 2, 6, 8, 9, 10, 12, 13]]
    weights = concat_list([i.get_weights() for i in layer])
    layer_t = model.get_layer(index = 3)
    assert [i.shape for i in weights] == [i.shape for i in layer_t.get_weights()]
    layer_t.set_weights(weights)
    save_model(model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/conv_best_model')
    '''

def init_sep_model():
    lm_model = build_lm_model_gelu.build_model(is_training=False)
    lm_model.summary()
    model = build_access_model_sep.build_model(is_training=False)
    model.summary()
    if load_model(lm_model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/pre_best_model') == -1:
        print('lm model weights not exist!')
        exit(-1)
    # print(lm_model.get_layer(index = 1).get_weights()[0].shape)
    # print(model.get_layer(index=2).get_weights()[0].shape)
    model.get_layer(index = 2).set_weights([lm_model.get_layer(index = 1).get_weights()[0]])
    save_model(model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/fine_best_model')
    layer_param = compute_param(lm_model.get_layer(index = 1).get_weights()[0])
    print(f'{layer_param} parameters have been set')
    '''
    layer = [lm_model.get_layer(index = i) for i in [1, 2, 6, 8, 9, 10, 12, 13]]
    weights = concat_list([i.get_weights() for i in layer])
    layer_t = model.get_layer(index = 3)
    assert [i.shape for i in weights] == [i.shape for i in layer_t.get_weights()]
    layer_t.set_weights(weights)
    save_model(model, root_path + '/data/fine_tune/input' + '256' + '/checkpoint0/startpoint')
    '''
    
if __name__ == '__main__':
    # init_conv_model()
    # init_pre_model()
    # init_feature_model()
    init_sep_model()