#!/usr/bin/env python3

import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cdist
import sys
from time import time

def strip(x):
    return x.strip()


embedding = KeyedVectors.load('../data/ASBC_embedding.bin')
with open('../data/ChineseStopWords_CT.txt') as stopwords_file:
    stopwords = map(strip,  stopwords_file.readlines())


def sent2vec(sent):
    # sent is a list of words
    
    embeds = []
    for word in sent:
        if word in embedding:
            if word not in stopwords:
                embeds.append(embedding[word])
    if embeds == []:
        return None

    vec = np.mean(embeds, axis=0)
    return vec


def array2tuple(array):
    return tuple(array.flatten().tolist())

    
def reviews2vec_review(reviews):
    vec_review = {}
    for review in reviews:
        words = review.split('|')
        vec = sent2vec(words)
        if vec is None:
            continue
        vec_review[array2tuple(vec)] = review
    return vec_review


start = time()
# TODO: noises (review, +1) and (review, -1)
review_pol = {}
with open('../data/polarity_review_seg.txt') as seg_file:
    for line in seg_file:
        pol, review = line.strip().split(' ')
        pol = int(pol)
        review_pol[review] = pol

vec_review_25w = reviews2vec_review(review_pol)
delta = time() - start
print('finish reading polarity_review, with time %f' % delta)
sys.stdout.flush()


start = time()
review_info = {}
with open('../data/aspect_review_seg.txt') as seg_file:
    lines = seg_file.readlines()
    num = len(lines) // 4
    for i in range(num):
        id_, review, pos, neg = map(strip, lines[4*i : 4*(i+1)])
        id_ = int(id_)
        pos = pos.split()
        neg = neg.split()
        review_info[review] = (pos, neg)

vec_review_200 = reviews2vec_review(review_info)
delta = time() - start
print('finish reading aspect_review, with time %f' % delta)
sys.stdout.flush()


def closest(A, B):
    dim = len(B)
    mat = cdist(A, B, metric='cosine')
    idx = np.argmin(mat)
    x, y = idx // dim, idx % dim
    return x, y, mat[x, y]


# TODO: move sentence from polarity review to aspect review, and assign the aspect to the sentence
rounds = 10
for i in range(rounds):
    # start = time()
    vecs_25w = list(vec_review_25w.keys())
    vecs_200 = list(vec_review_200.keys())
    x, y, dist = closest(vecs_25w, vecs_200)
    # delta = time() - start
    print(vec_review_25w[vecs_25w[x]])
    print(vec_review_200[vecs_200[y]])
    # print(dist)
    # print('time %f' % delta)
    sys.stdout.flush()

    del vec_review_25w[vecs_25w[x]]
