import numpy as np
import sys,datetime, os
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()
       

def split_and_pad(name, X_seqs, seq_len, verbose):
    if name == 'bilstm': 
        # an example is 'SATGCE', we have 4 x_pre and x_post
        x_pre = [
            X_seq[:i] for X_seq in X_seqs for i in range(1, len(X_seq) - 1)#delete start and end of seq (22,23)
        ]
        x_post = [
            X_seq[i + 1:] for X_seq in X_seqs for i in range(1, len(X_seq) - 1)
        ]
        y = np.array([
            X_seq[i] for X_seq in X_seqs for i in range(1, len(X_seq) - 1)
        ])
        #if verbose > 1:
        #    tprint('splitted context samples: {}'.format(len(x_pre)))
        x_pre = pad_sequences(
            x_pre, maxlen = seq_len - 1,
            dtype = 'int32', padding = 'pre', truncating = 'pre', value = 0 # pad or trucate at the prepositon for the previous context
        )
        x_post = pad_sequences(
            x_post, maxlen=seq_len - 1,
            dtype='int32', padding='post', truncating='post', value=0   # pad or trucate at the postpositon for the post context
        )
#        if verbose > 1:
#            tprint('Flipping...')
        x_post = np.flip(x_post, axis = 1)
        '''
        For example, we generate a context of 'SATGCE' which is 'T' with 'SA' and 'ECG'.
        with padding_size = 4 we can get 'T' with 'PPSA' and 'PECG', which based on the intuition that dna is
        order insensitive.
        '''
        X = [ x_pre, x_post ]

        #if verbose > 1:
        #    tprint('done splitting and padding')
        return X, y
    else:
        return None, None


def load_data(File):
    seqs = np.load(File, allow_pickle = True)
    seqs = seqs.tolist()
    return seqs


if __name__ == '__main__':
    #max_len = 500
    max_len = int(sys.argv[1])
    batchNum = int(sys.argv[2])
    root_path = os.environ['d']
    samples_num = []

    seqs = load_data(root_path + '/data/post_train/dna_train_source.npy')
    for index in range(0, len(seqs), batchNum): 
        #print("*************" + str(index) + "*****************")
        end_index = index + batchNum
        if end_index > len(seqs):
            end_index = len(seqs)   # [index, end_index)
        x, y = split_and_pad('bilstm', seqs[index:end_index], max_len, 2)
        np.save(root_path + '/data/post_train/input' + str(max_len) + '/train_x_' + str(index // batchNum), x)
        np.save(root_path + '/data/post_train/input' + str(max_len) + '/train_y_' + str(index // batchNum), y)
        tprint('%-8dsplitted context samples saved to %s' %
        (len(y), 'train_' + str(index // batchNum)))
        samples_num.append(len(y))
        #print(X[0].shape)
        del x, y
    del seqs

    # print("----------------validation----------------------")
    seqs = load_data(root_path + '/data/post_train/dna_valid_source.npy')
    x, y = split_and_pad('bilstm', seqs, max_len, 2)
    np.save(root_path + '/data/post_train/input' + str(max_len) + '/valid_x_0', x)
    np.save(root_path + '/data/post_train/input' + str(max_len) + '/valid_y_0', y)
    tprint('%-8dsplitted context samples saved to %s' %
        (len(y), 'valid_0'))
    samples_num.append(len(y))
    # print(X[0].shape)
    del seqs, x, y
    
    np.save(root_path + '/data/post_train/input' + str(max_len) + '/train_valid_num', samples_num)
    # file format: x = [[seq1, seq2...], [seq1, seq2...]], y = [target1, target2...]
