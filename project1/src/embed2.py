#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from gensim.models import KeyedVectors
from sklearn import svm
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from embed import *
from time import time


aspect2idx = {
    '服務': 0,
    '環境': 1,
    '價格': 2,
    '交通': 3,
    '餐廳': 4
}


start = time()
with open('../data/polarity_review_aspect_1_id.txt') as in_file:
    line = in_file.readline()
    l = 0
    
    # L = number of instances.
    # A review with 3 aspects counts as 3 instances.
    # L = 165832
    L = 114933
    X = np.zeros([L, 6])
    while line != '':
        id_ = int(line.strip())

        line = in_file.readline()
        review = line.strip()
        # words = review.split('|')
        # vec = sent2vec(words)

        line = in_file.readline()
        pos = line.strip().split()

        line = in_file.readline()
        neg = line.strip().split()
        
        onehot = np.zeros(6)
        onehot[0] = id_
        
        for aspect in pos:
            onehot[aspect2idx[aspect] + 1] = +1
        for aspect in neg:
            onehot[aspect2idx[aspect] + 1] = -1

        X[l] = onehot
        l += 1
        line = in_file.readline()

    print(X.shape)
    
np.save('X.npy', X)
delta = time() - start
info('finish creating X', delta)
