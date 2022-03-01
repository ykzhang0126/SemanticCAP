import os
import sys
sys.path.append(os.environ['d'])
import build_access_model_conv
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow_addons.shared import *

root_path = os.environ['d']
seq_len = int(sys.argv[1])
max_len = int(sys.argv[2])
max_epoch = int(sys.argv[3])
batch_size = int(sys.argv[4])
tmode = 'fine_tune'

def Data_generator(DS_xb, DS_xa, DS_in, DS_y, batchSize=256):
    # generate train data in batchs in random order, epoch after epoch
    # Len = DS[0].shape[0]
    Len = DS_y.shape[0]
    # print("************"+str(Len))
    BatchesNum = int(Len / batchSize)  # we left some last data
    DSIndex = list(range(BatchesNum * batchSize))
    SEQIndex = [i for i in range((768 - seq_len) // 2, (768 + seq_len) // 2)]
    while True:
        # np.random.shuffle(DSIndex)
        for batchNum in range(BatchesNum):
            start = batchNum * batchSize
            end = start + batchSize
            BatchDSXpre = DS_xb[start:end, (768 - seq_len) // 2: (768 + seq_len) // 2]
            BatchDSXpost = DS_xa[start:end, (768 - seq_len) // 2: (768 + seq_len) // 2]
            BatchDSXin = DS_in[start:end, (768 - seq_len) // 2: (768 + seq_len) // 2]
            BatchDSy = DS_y[start:end]
            yield [BatchDSXpre, BatchDSXpost, BatchDSXin], BatchDSy

def train():
    OutputDir = root_path + '/data/' + tmode + '/output' + str(max_len) + '/'
    if not os.path.exists(OutputDir + '/checkpoint/'):
        os.makedirs(OutputDir + '/checkpoint/')
    global seq_len
    seq_len_bk = seq_len
    seq_len = 768
    train_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_pre.npy', mode='r')
    train_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_pre.npy', mode='r',
                         shape=(train_xb.shape[0] // seq_len // max_len, seq_len, max_len))
    train_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_post.npy', mode='r')
    train_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_post.npy', mode='r',
                         shape=(train_xa.shape[0] // seq_len // max_len, seq_len, max_len))
    train_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_in.npy', mode='r')
    train_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_in.npy', mode='r',
                         shape=(train_xi.shape[0] // seq_len, seq_len))
    train_y = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_y.npy', mode='r')

    valid_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_pre.npy', mode='r')
    valid_xb = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_pre.npy', mode='r',
                         shape=(valid_xb.shape[0] // seq_len // max_len, seq_len, max_len))
    valid_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_post.npy', mode='r')
    valid_xa = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_post.npy', mode='r',
                         shape=(valid_xa.shape[0] // seq_len // max_len, seq_len, max_len))
    valid_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_in.npy', mode='r')
    valid_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_in.npy', mode='r',
                         shape=(valid_xi.shape[0] // seq_len, seq_len))
    valid_y = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_y.npy', mode='r')
    seq_len = seq_len_bk

    vocab_size = 4 + 2

    model = build_access_model_conv.build_model(seq_len, max_len, vocab_size)
    '''
    with open('vrb', 'w') as fout:
        for i in model.variables:
            fout.write(str(i) + '\n\n\n\n')
    '''
    model.summary()

    print('Try to load trained model...')
    start_model = root_path + '/data/' + tmode + '/input' + str(max_len) + '/' + '/checkpoint0/startpoint' + str(seq_len)
    load_model(model, start_model)
    adam = Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=OutputDir + '/checkpoint/', histogram_freq=1)
    reduceLRpatience = 2
    earlyStopPatience = reduceLRpatience * 2
    reduceLR = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.2,
                                 patience=reduceLRpatience)
    earlystopping = EarlyStopping(monitor='val_loss',
                                  patience=earlyStopPatience,
                                  verbose=1,
                                  mode='auto')
    model_save = Model_save(model, OutputDir + '/checkpoint/')
    # save_model(model, OutputDir + '/checkpoint/' + '/[-1]')

    trainLen = train_y.shape[0]
    trainBatchesNum = int(trainLen / batch_size)
    validLen = valid_y.shape[0]
    validBatchesNum = int(validLen / batch_size)
    
    '''
    stps = max_epoch * trainBatchesNum
    td = Data_generator(train_xb, train_xa, train_xi, train_y, batch_size)
    vd = Data_generator(valid_xb, valid_xa, valid_xi, valid_y, batch_size)
    i = 0
    intv = 4
    for epc in range(max_epoch):
        losses = []
        accs = []
        val_losses = []
        val_accs = []
        for bch in range(trainBatchesNum):
            tx, ty = next(td)
            i += 1
            t_his = model.train_on_batch(tx, ty, return_dict = True)
            losses.append(t_his['loss'])
            accs.append(t_his['accuracy'])
            if i % intv == 0:
                print("[train step %d/%d] loss: %.3f, acc: %.3f" % (i, stps, sum(losses) / (intv), sum(accs) / (intv)),
                    flush = True)
                losses = []
                accs = []
        for bch in range(validBatchesNum):
            vx, vy = next(vd)
            v_his = model.test_on_batch(vx, vy, return_dict = True)
            val_losses.append(v_his['loss'])
            val_accs.append(v_his['accuracy'])
        print("[epoch %d] val_loss: %.3f, val_acc: %.3f" % (epc + 1, sum(val_losses) / validBatchesNum, sum(val_accs) / validBatchesNum),
            flush=True)
        save_model(model, OutputDir + '/checkpoint/' + '/[' + str(epc) + ']' + str(sum(val_losses) / validBatchesNum))
    '''
    history_callback = model.fit_generator(
        Data_generator(train_xb, train_xa, train_xi, train_y, batch_size),
        steps_per_epoch=trainBatchesNum,
        epochs=max_epoch,
        verbose=1,
        callbacks=[model_save, tensorboard, reduceLR, earlystopping],
        validation_data=Data_generator(valid_xb, valid_xa, valid_xi, valid_y, batch_size),
        validation_steps=validBatchesNum
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
