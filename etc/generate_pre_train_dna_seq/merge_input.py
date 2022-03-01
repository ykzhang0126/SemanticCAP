import numpy as np
import os
import sys

def load_data(File):
    seqs = np.load(File, allow_pickle = True)
    seqs = seqs.tolist()
    return seqs

def merge_input100(dirPrefix, pmode, samples_num):
    maxLen = int(sys.argv[1])
    #seqsNum = 32194
    #samples_num = [6375003,6375000,6375000,6375000,6374212,6361770,2790768]
    X_pre = np.memmap(dirPrefix+"/" + pmode + "_x_pre.npy",mode='w+', shape=(sum(samples_num),maxLen - 1))
    X_post = np.memmap(dirPrefix+"/" + pmode + "_x_post.npy",mode='w+',shape=(sum(samples_num),maxLen - 1))
    Y_all = np.memmap(dirPrefix+"/" + pmode + "_y.npy",mode='w+',shape=(sum(samples_num)))

    index = 0
    start = 0
    fileName = dirPrefix+"/" + pmode + "_x_" + str(index) + ".npy"
    while os.path.isfile(fileName):
        #" + pmode + "X = np.lib.format.open_memmap(dirPrefix+"_X_"+str(index*batchNum)+'.npy')
        X = np.load(dirPrefix+"/" + pmode + "_x_" + str(index) + ".npy")
        #" + pmode + "Y = np.lib.format.open_memmap(dirPrefix+"_y_"+str(index*batchNum)+'.npy')
        Y = np.load(dirPrefix+"/" + pmode + "_y_" + str(index) + ".npy")
        if samples_num[index] != X.shape[1]:
            print("error: " + pmode + "" + str(index))
            exit(-1)
        #print(start,start+samples_num[index],samples_num[index])
        X_pre[start:start+samples_num[index]] = X[0][:]
        X_post[start:start+samples_num[index]] = X[1][:]
        Y_all[start:start+samples_num[index]] = Y[:]
        del X, Y
        start += samples_num[index]	
        index+=1
        fileName = dirPrefix+"/" + pmode + "_x_" + str(index) + ".npy"

    X_pre.flush()
    X_post.flush()
    Y_all.flush()

    return [X_pre,X_post],Y_all


if __name__ == '__main__':
    root_path = os.environ['d']
    
    samples_num = load_data(root_path + '/data/pre_train/input' + sys.argv[1] + '/train_valid_num.npy')
    X,Y = merge_input100(root_path + '/data/pre_train/input' + sys.argv[1], 'train', samples_num[0: len(samples_num) - 1])
    print("total: %d samples for train, " % Y.shape[0], end = '')
    X,Y = merge_input100(root_path + '/data/pre_train/input' + sys.argv[1], 'valid', samples_num[len(samples_num) - 1: len(samples_num)])
#   print(X[0].shape[0])
    print("%d samples for valid" % Y.shape[0])


    #print('merge ok')
    #print(X.shape)
    #print(X[0])
    #print(X[1])
    #print(Y)

