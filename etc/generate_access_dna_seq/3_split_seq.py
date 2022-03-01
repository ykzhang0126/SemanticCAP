import numpy as np
import sys,datetime, os

def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()
       

def split_and_pad(X_seqs, seq_len):
    # an example is 'SATGCE', we have 4 x_pre and x_post
    x_pre = [[
        X_seq[i - seq_len:i]  for i in range(seq_len, len(X_seq) - 1 - seq_len)#delete start and end of seq (22,23)
    ] for X_seq in X_seqs ]
    # print(np.array(x_pre).shape)
    x_post = [[
        X_seq[i + 1: i + 1 + seq_len]  for i in range(seq_len, len(X_seq) - 1 - seq_len)#delete start and end of seq (22,23)
    ] for X_seq in X_seqs ]
    x_in = [[
        X_seq[i]  for i in range(seq_len, len(X_seq) -1 - seq_len)#delete start and end of seq (22,23)
    ] for X_seq in X_seqs ]
    y = [
        X_seq[-1]
        for X_seq in X_seqs ]

    x_post = np.flip(x_post, axis = 2)
    '''
    For example, we generate a context of 'SATGCE' which is 'T' with 'SA' and 'ECG'.
    with padding_size = 4 we can get 'T' with 'PPSA' and 'PECG', which based on the intuition that dna is
    order insensitive.
    '''
    #if verbose > 1:
    #    tprint('done splitting and padding')
    return x_pre, x_post, x_in, y
    # 1000, 768, 256

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

    for tmode in ('test', 'valid', 'train'):
        seqs = load_data(root_path + '/data/fine_tune/dna_' + tmode + '_source.npy')
        for index in range(0, len(seqs), batchNum): 
            #print("*************" + str(index) + "*****************")
            end_index = index + batchNum
            if end_index > len(seqs):
                end_index = len(seqs)   # [index, end_index)
            x_pre, x_post, x_in, y = split_and_pad(seqs[index:end_index], max_len)
            # print(len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]))
            np.save(root_path + '/data/fine_tune/input' + str(max_len) + '/' + tmode + '_x_pre_' + str(index // batchNum), x_pre)
            np.save(root_path + '/data/fine_tune/input' + str(max_len) + '/' + tmode + '_x_post_' + str(index // batchNum), x_post)
            np.save(root_path + '/data/fine_tune/input' + str(max_len) + '/' + tmode + '_x_in_' + str(index // batchNum), x_in)
            np.save(root_path + '/data/fine_tune/input' + str(max_len) + '/' + tmode + '_y_' + str(index // batchNum), y)
            tprint('%-8ddna samples saved to %s' %
            (len(y), tmode + '_' + str(index // batchNum)))
            samples_num.append(len(y))
            #print(X[0].shape)
            del x_pre, x_post, x_in, y
        del seqs

    np.save(root_path + '/data/fine_tune/input' + str(max_len) + '/train_valid_test_num', samples_num)
    # file format: x = [[seq1, seq2...], [seq1, seq2...]], y = [target1, target2...]
