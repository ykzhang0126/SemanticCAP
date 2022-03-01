import os
import sys
sys.path.append(os.environ['d'])
from tensorflow_addons.shared import *
import build_access_model_conv
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
from sklearn import metrics

root_path = '/home/ykzhang/data_process/'
seq_len = 768
max_len = 256
max_epoch = 32
batch_size = 1024

root_path = os.environ['d']
seq_len = int(sys.argv[1])
max_len = int(sys.argv[2])
max_epoch = int(sys.argv[3])
batch_size = int(sys.argv[4])
feature_len = 192
tmode = 'fine_tune'


def Data_generator(DS_xf, DS_x, DS_y, batchSize=256):
    # generate train data in batchs in random order, epoch after epoch
    Len = DS_y.shape[0]
    BatchesNum = Len // batchSize  # we left some last data
    while True:
        for batchNum in range(BatchesNum):
            start = batchNum * batchSize
            end = start + batchSize
            BatchDSXf = DS_xf[start:end]
            BatchDSX = DS_x[start:end]
            BatchDSY = DS_y[start:end]
            yield [BatchDSXf, BatchDSX], BatchDSY
            del BatchDSXf, BatchDSX, BatchDSY

def train():
    OutputDir = root_path + '/data/' + tmode + '/output' + str(max_len) + '/'
    if not os.path.exists(OutputDir + '/checkpoint/'):
        os.makedirs(OutputDir + '/checkpoint/')

    train_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_in.npy', mode='r')
    train_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_x_in.npy', mode='r',
                         shape=(train_xi.shape[0] // seq_len, seq_len))
    train_y = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_y.npy', mode='r')
    valid_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_in.npy', mode='r')
    valid_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_x_in.npy', mode='r',
                         shape=(valid_xi.shape[0] // seq_len, seq_len))
    valid_y = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_y.npy', mode='r')
    train_xf = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/train_xf.npy',
                         dtype='float32', mode='r', shape=(train_xi.shape[0], seq_len, feature_len))
    valid_xf = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/valid_xf.npy',
                         dtype='float32', mode='r', shape=(valid_xi.shape[0], seq_len, feature_len))

    model = build_access_model_conv.build_model(is_training=True)
    '''
    with open('vrb', 'w') as fout:
        for i in model.variables:
            fout.write(str(i) + '\n\n\n\n')
    '''
    model.summary()
    print('Try to load trained model...')
    start_model = root_path + '/data/' + tmode + '/input' + str(max_len) + '/' + '/checkpoint0/fine_best_model'
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
    '''
    trainLen = train_y.shape[0]
    trainBatchesNum = int(trainLen / batch_size)
    validLen = valid_y.shape[0]
    validBatchesNum = int(validLen / batch_size)

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
        Data_generator(DS_xf=train_xf, DS_x=train_xi, DS_y=train_y, batchSize=batch_size),
        steps_per_epoch=train_y.shape[0] // batch_size,
        epochs=max_epoch,
        verbose=1,
        callbacks=[model_save, tensorboard, reduceLR, earlystopping],
        validation_data=Data_generator(DS_xf=valid_xf, DS_x=valid_xi, DS_y=valid_y, batchSize=batch_size),
        validation_steps=valid_y.shape[0] // batch_size
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


def test(method):
    test_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/test_x_in.npy', mode='r')
    test_xi = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/test_x_in.npy', mode='r',
                        shape=(test_xi.shape[0] // seq_len, seq_len))
    test_y = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/test_y.npy', mode='r')
    test_xf = np.memmap(root_path + '/data/' + tmode + '/input' + str(max_len) + '/test_xf.npy',
                        dtype='float32', mode='w+', shape=(test_xi.shape[0], seq_len, feature_len))
    model = build_access_model_conv.build_model(is_training=False)
    model.summary()
    print('Try to load trained model...')
    start_model = root_path + '/data/' + tmode + '/output' + str(max_len) + '/' + method + '/fine_best_model'
    load_model(model, start_model)

    datag = Data_generator(DS_xf=test_xf, DS_x=test_xi, DS_y=test_y, batchSize=batch_size)
    pred_y = model.predict_generator(generator=datag, steps=test_y.shape[0] // batch_size).reshape(-1)
    pred_y = [1 if i > 0.5 else 0 for i in pred_y]
    test_y = test_y[0: test_y.shape[0] // batch_size * batch_size]
    print('Calculating AUC...')
    print(pred_y, test_y)
    auroc = metrics.roc_auc_score(test_y, pred_y)
    auprc = metrics.average_precision_score(test_y, pred_y)
    precision = metrics.precision_score(test_y, pred_y)
    print(precision, auroc, auprc)


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train()
    # test('checkpoint')
