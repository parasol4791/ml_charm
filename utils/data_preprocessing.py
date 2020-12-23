import numpy as np

def vectorize(seqs, dimension):
    '''Vectorize integer sequences, and One-hot encode them.
       Seqs is an array of lists, dimension is typically a number of categories
       in an individual sequence (number of words in classification problem)'''
    res = np.zeros( (len(seqs), dimension) )
    for i, seq in enumerate(seqs):
        res[i,seq] = 1
    return res


def decodeSequence(dataset, seq):
    '''Decodes numerical sequence from a dataset'''
    idx = dataset.get_word_index()
    rev_idx = { value : key for key, value in idx.items() }
    for i in seq:
        if not i-3 in rev_idx:
            print(i)
    return [rev_idx.get(idx-3, '?') for idx in seq]
