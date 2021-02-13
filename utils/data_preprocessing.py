import os
import numpy as np


def vectorize(seqs, dimension):
    """Vectorize integer sequences, and One-hot encode them.
       Seqs is an array of lists, dimension is typically a number of categories
       in an individual sequence (number of words in classification problem)"""
    res = np.zeros((len(seqs), dimension))
    for i, seq in enumerate(seqs):
        res[i, seq] = 1
    return res


def decodeSequence(dataset, seq):
    """Decodes numerical sequence from a dataset"""
    idx = dataset.get_word_index()
    rev_idx = {value: key for key, value in idx.items()}
    for i in seq:
        if i-3 not in rev_idx:
            print(i)
    return [rev_idx.get(idx-3, '?') for idx in seq]


def shuffle(array1, array2):
    """Randomly shuffles 2 numpy arrays"""
    sh = array1.shape[0]
    if sh != array2.shape[0]:
        raise AssertionError('Array shapes are different: {} vs {}'.format(sh, array2.shape[0]))
    indices = np.arange(sh)
    np.random.shuffle(indices)
    array1 = array1[indices]
    array2 = array2[indices]
    return array1, array2


def embeddings_GloVe(tokens='6B', dim=100):
    """Global Vectors for word representaion from
     https://nlp.stanford.edu/projects/glove/
     6B tokens, 400k vocabulary"""
    if tokens != '6B':
        raise ValueError('Only GloVe with 6B tokens is currently supported. Attempted {}'.format(tokens))
    if dim not in [50, 100, 200, 300]:
        raise ValueError('Only embedding vectors with dimentions 50, 100, 200, 300 are supported. Attempted {}'.format(dim))
    data_dir = os.environ['DATASETS_DIR']
    glove_dir = os.path.join(data_dir, 'glove', 'glove.{}.{}d.txt'.format(tokens, dim))  # like, glove.6B.100d.txt

    # Extract embedding coefficients
    embed_idx = {}
    with open(glove_dir) as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            coeffs = np.asarray(vals[1:], dtype='float32')
            embed_idx[word] = coeffs  # map of word : coefficients
        print('Found {} word vectors'.format(len(embed_idx)))

    return embed_idx
