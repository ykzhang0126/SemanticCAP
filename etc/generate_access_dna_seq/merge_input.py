import numpy as np
import os
import sys

maxLen = int(sys.argv[1])
seq_len = int(sys.argv[2])

def load_data(File):
    seqs = np.load(File, allow_pickle = True)
    seqs = seqs.tolist()
    return seqs

def merge_input(dirPrefix, pmode, samples):
    #seqsNum = 32194
    #samples_num = [6375003,6375000,6375000,6375000,6374212,6361770,2790768]
    X_pre = np.memmap(dirPrefix+"/tmp/" + pmode + "_x_pre.npy",mode='w+', shape=(sum(samples), seq_len, maxLen))
    X_post = np.memmap(dirPrefix+"/tmp/" + pmode + "_x_post.npy",mode='w+',shape=(sum(samples), seq_len, maxLen))
    X_in = np.memmap(dirPrefix+"/tmp/" + pmode + "_x_in.npy",mode='w+',shape=(sum(samples), seq_len))
    Y_all = np.memmap(dirPrefix+"/tmp/" + pmode + "_y.npy",mode='w+',shape=(sum(samples)))

    index = 0
    start = 0
    fileName = dirPrefix+"/" + pmode + "_y_" + str(index) + ".npy"
    while os.path.isfile(fileName):
        print(pmode + ' ' + str(index) + '(' + str(samples[index]) + ')' + ' is being processed')
        x_pre = np.load(dirPrefix+"/" + pmode + "_x_pre_" + str(index) + ".npy")
        x_post = np.load(dirPrefix+"/" + pmode + "_x_post_" + str(index) + ".npy")
        x_in = np.load(dirPrefix+"/" + pmode + "_x_in_" + str(index) + ".npy")
        y = np.load(dirPrefix+"/" + pmode + "_y_" + str(index) + ".npy")
        if samples[index] != y.shape[0]:
            print("error: " + pmode + " " + str(index))
            exit(-1)
        #print(start,start+samples_num[index],samples_num[index])
        X_pre[start:start+samples[index]] = x_pre
        X_post[start:start+samples[index]] = x_post
        X_in[start:start+samples[index]] = x_in
        Y_all[start: start + samples[index]] = y
        del x_pre, x_post, x_in, y
        start += samples[index]	
        index += 1
        fileName = dirPrefix + "/" + pmode + "_y_" + str(index) + ".npy"

    X_pre.flush()
    X_post.flush()
    X_in.flush()
    Y_all.flush()

    return Y_all.shape[0]


if __name__ == '__main__':
    root_path = os.environ['d']
    
    samples_num = load_data(root_path + '/data/fine_tune/input' + sys.argv[1] + '/train_valid_test_num.npy')
    pos = []
    for i in range(len(samples_num)):
        if (samples_num[i] < 3000):
            pos.append(i + 1)

    n = merge_input(root_path + '/data/fine_tune/input' + sys.argv[1], 'test', samples_num[0: pos[0]])
    print("%d samples for test" % n)
    n = merge_input(root_path + '/data/fine_tune/input' + sys.argv[1], 'valid', samples_num[pos[0]: pos[1]])
    print("%d samples for valid" % n)
    n = merge_input(root_path + '/data/fine_tune/input' + sys.argv[1], 'train', samples_num[pos[1]: pos[2]])
    print("%d samples for train" % n)

    #print('merge ok')
    #print(X.shape)
    #print(X[0])
    #print(X[1])
    #print(Y)

