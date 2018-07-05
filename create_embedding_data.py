# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:14:32 2018

@author: dhanu
"""


import pickle
import bcolz
import numpy as np

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'Embedding/6B.300d.dat', mode='w')

with open(f'Embedding/glove.6B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400001, 300)), rootdir=f'Embedding/6B.300d.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'Embedding/6B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'Embedding/6B.300_idx.pkl', 'wb'))
