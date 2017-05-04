#!/usr/bin/env python3

import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cdist
import sys
from time import time


def info(msg, t):
    print('%s, with time %f' % (msg, t))
    sys.stdout.flush()


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


if __name__ == '__main__':
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
    info('finish reading polarity_review', delta)

    start = time()
    review_info = {}
    with open('../data/aspect_review_seg.txt') as seg_file:
        lines = seg_file.readlines()
        num = len(lines) // 4
        for i in range(num):
            id_, review, pos, neg = map(strip, lines[4*i : 4*(i+1)])
            id_ = int(id_)
            # pos = pos.split()
            # neg = neg.split()
            review_info[review] = (pos, neg)

    vec_review_200 = reviews2vec_review(review_info)
    delta = time() - start
    info('finish reading aspect_review', delta)

    vecs25w = list(vec_review_25w.keys())
    vecs200 = list(vec_review_200.keys())
    dim = len(vecs25w), len(vecs200)
    start = time()
    cosine = cdist(vecs25w, vecs200, metric='cosine')
    delta = time() - start
    info('finish creating cosine matrix', delta)

    similar = cosine.argsort(1)[:, 0]
    # sort[i]: most similar to least similar aspect_reviews to i-th polarity_review
    start = time()
    with open('../data/polarity_review_aspect.txt', 'w') as out_file:
        for i, j in enumerate(similar):
            vec_i, vec_j = vecs25w[i], vecs200[j]
            review_i, review_j = vec_review_25w[vec_i], vec_review_200[vec_j]
            pos, neg = review_info[review_j]
            
            # TODO: decide which aspects to take
            # maybe dont take all, or also take opposite ones
            if review_pol[review_i] > 0:
                if pos != '':
                    out_file.write(review_i + '\n')
                    out_file.write(pos + '\n')
                    out_file.write('\n')
                    out_file.write('%.6f\n' % cosine[i, j])
            else:
                if neg != '':
                    out_file.write(review_i + '\n')
                    out_file.write('\n')
                    out_file.write(neg + '\n')
                    out_file.write('%.6f\n' % cosine[i, j])
                    
    delta = time() - start
    info('finish output file', delta)
