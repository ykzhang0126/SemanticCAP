# https://blog.csdn.net/qq_40128284/article/details/109997328
import sys
import os
sys.path.append(os.environ['d'])

import tensorflow_addons as ta
from tensorflow_addons.shared import *
import build_lm_model_gelu
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

root_path = os.environ['d']
seq_len = int(sys.argv[1])
tmode = sys.argv[4]
maxEpoch = int(sys.argv[2])
batchSize = int(sys.argv[3])
lrr = float(sys.argv[5])


def Data_generator(DS_xb, DS_xa, DS_y, batchSize=256):
    # generate train data in batchs in random order, epoch after epoch
    # Len = DS[0].shape[0]
    Len = DS_y.shape[0]
    # print("************"+str(Len))
    BatchesNum = int(Len / batchSize)  # we left some last data
    DSIndex = list(range(BatchesNum * batchSize))
    while True:
        # np.random.shuffle(DSIndex)
        for batchNum in range(BatchesNum):
            start = batchNum * batchSize
            end = start + batchSize
            BatchDSXpre = DS_xb[DSIndex[start:end]]
            BatchDSXpost = DS_xa[DSIndex[start:end]]
            BatchDSy = DS_y[DSIndex[start:end]]
            yield [BatchDSXpre, BatchDSXpost], BatchDSy

def load_data(File):
    seqs = np.load(File, allow_pickle=True)
    seqs = seqs.tolist()
    return seqs


def train():
    OutputDir = root_path + '/data/' + tmode + '/output' + str(seq_len) + '/'
    if not os.path.exists(OutputDir + '/checkpoint/'):
        os.makedirs(OutputDir + '/checkpoint/')
    samples_num = load_data(root_path + '/data/' + tmode + '/input' + str(seq_len) + '/train_valid_num.npy')
    train_samples_num = sum(samples_num[0: len(samples_num) - 1])
    valid_samples_num = sum(samples_num[len(samples_num) - 1: len(samples_num)])

    train_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(seq_len) + '/train_x_pre.npy', mode='r',
                         shape=(train_samples_num, seq_len - 1))
    train_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(seq_len) + '/train_x_post.npy', mode='r',
                         shape=(train_samples_num, seq_len - 1))
    train_y = np.memmap(root_path + '/data/' + tmode + '/input' + str(seq_len) + '/train_y.npy', mode='r',
                        shape=(train_samples_num,))
    # print(train_y)
    valid_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(seq_len) + '/valid_x_pre.npy', mode='r',
                         shape=(valid_samples_num, seq_len - 1))
    valid_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(seq_len) + '/valid_x_post.npy', mode='r',
                         shape=(valid_samples_num, seq_len - 1))
    valid_y = np.memmap(root_path + '/data/' + tmode + '/input' + str(seq_len) + '/valid_y.npy', mode='r',
                        shape=(valid_samples_num,))

    # trainX,trainY,testX,testY = load_data()
    # seq_len = trainX[0].shape[1]
    vocab_size = 4 + 2

    model = build_lm_model_gelu.build_model(seq_len - 1, vocab_size, is_training = True)
    model.summary()
    
    print('Try to load trained model...')
    start_model = root_path + '/data/' + tmode + '/input' + str(seq_len) + '/' + '/checkpoint0/validBestModel'
    load_model(model, start_model)
    #opt = Adam(learning_rate=lrr, beta_1=0.9, beta_2=0.98, epsilon=1e-08)
    #opt = ta.optimizers.LAMB(learning_rate = lrr, epsilon = 1e-8)
    opt = ta.optimizers.Lookahead(ta.optimizers.RectifiedAdam(learning_rate = lrr))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer = opt, metrics = ['accuracy'])

    # trainBestModel = OutputDir + '/checkpoint/trainBestModel'
    # valiBestModel = OutputDir + '/checkpoint/validBestModel'
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=OutputDir + '/checkpoint/', histogram_freq=1)
    #checkpointer = ModelCheckpoint(filepath=valiBestModel,save_weights_only=True,verbose=1,save_best_only=True)
    reduceLRpatience = 2
    # earlyStopPatience = 6
    reduceLR = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.2,
                                 patience=reduceLRpatience)
    #earlystopping = EarlyStopping(monitor='val_loss',
    #                              patience=earlyStopPatience,
    #                              verbose=1,
    #                              mode='auto')
    model_save = Model_save(model, OutputDir + '/checkpoint/')

    trainLen = train_y.shape[0]
    trainBatchesNum = int(trainLen / batchSize)
    validLen = valid_y.shape[0]
    validBatchesNum = int(validLen / batchSize)
    
    history_callback = model.fit_generator(
        Data_generator(train_xb, train_xa, train_y, batchSize),
        steps_per_epoch = trainBatchesNum,
        epochs = maxEpoch,
        verbose = 1,
        callbacks = [model_save, reduceLR],
        validation_data = Data_generator(valid_xb, valid_xa, valid_y, batchSize),
        validation_steps= validBatchesNum,
        #workers = 8,
        #use_multiprocessing = True
        # 0 is nothing, 1 is progress bar, 2 is a record for each epoch
    )

    log_loss = OutputDir + '/checkpoint/' + 'loss.txt'
    log_acc = OutputDir + '/checkpoint/' + 'acc.txt'
    log_val_loss = OutputDir + '/checkpoint/' + 'valiLoss.txt'
    log_val_acc = OutputDir + '/checkpoint/' + 'valiAcc.txt'
    loss_history = history_callback.history["loss"]
    acc_history = history_callback.history["accuracy"]
    val_loss_history = history_callback.history["val_loss"]
    val_acc_history = history_callback.history["val_accuracy"]
    np.savetxt(log_loss, loss_history, delimiter=",")
    np.savetxt(log_acc, acc_history, delimiter=",")
    np.savetxt(log_val_loss, val_loss_history, delimiter=",")
    np.savetxt(log_val_acc, val_acc_history, delimiter=",")

        
    

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train()
