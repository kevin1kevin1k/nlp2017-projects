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


def sent2vecs(sent):
    # sent is a list of words
    
    embeds = []
    for word in sent:
        if word in embedding:
            if word not in stopwords:
                embeds.append(embedding[word])
    if embeds == []:
        return None
    
    return embeds
    

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


aspect2idx = {
    '服務': 0,
    '環境': 1,
    '價格': 2,
    '交通': 3,
    '餐廳': 4
}


if __name__ == '__main__':
    start = time()
    # TODO: noises (review, +1) and (review, -1)
    review_pol = {}
    review_id = {}
    with open('../data/polarity_review_seg.txt') as seg_file:
        id_ = 0
        for line in seg_file:
            pol, review = line.strip().split(' ')
            pol = int(pol)
            review_pol[review] = pol
            review_id[review] = id_
            id_ += 1

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


    aspect_terms = []
    with open('../data/aspect_term_0505.txt') as aspect_term_file:
        for line in aspect_term_file:
            terms = line.strip().split()
            aspect_terms.append(terms)


    similar = cosine.argsort(1)[:, 0]
    # similar[i]: most similar to least similar aspect_reviews to i-th polarity_review
    start = time()
    with open('../data/polarity_review_aspect_1_id.txt', 'w') as out_file:
        for i, j in enumerate(similar):
            vec_i, vec_j = vecs25w[i], vecs200[j]
            review_i, review_j = vec_review_25w[vec_i], vec_review_200[vec_j]
            pos, neg = review_info[review_j]
            
            # TODO: decide which aspects to take
            # maybe dont take all, or also take opposite ones
            if review_pol[review_i] > 0:
                if pos != '':
                    min_cos = 2.0
                    for a in pos.split():
                        aspect = aspect2idx[a]
                        vecs_review = sent2vecs(review_i)
                        vecs_aspect = sent2vecs(aspect_terms[aspect])
                        if vecs_review is None or vecs_aspect is None:
                            continue
                        cos = np.min(cdist(vecs_review, vecs_aspect))
                        if cos < min_cos:
                            min_cos = cos
                            arg_min_cos = a
                    
                    if min_cos < 2:
                        out_file.write(str(review_id[review_i]) + '\n')
                        out_file.write(review_i + '\n')
                        out_file.write(arg_min_cos + '\n')
                        out_file.write('\n')
                        # out_file.write('%.6f\n' % cosine[i, j])
            else:
                if neg != '':
                    min_cos = 2.0
                    for a in neg.split():
                        aspect = aspect2idx[a]
                        vecs_review = sent2vecs(review_i)
                        vecs_aspect = sent2vecs(aspect_terms[aspect])
                        if vecs_review is None or vecs_aspect is None:
                            continue
                        cos = np.min(cdist(vecs_review, vecs_aspect))
                        if cos < min_cos:
                            min_cos = cos
                            arg_min_cos = a
                            
                    if min_cos < 2:
                        out_file.write(str(review_id[review_i]) + '\n')
                        out_file.write(review_i + '\n')
                        out_file.write('\n')
                        out_file.write(arg_min_cos + '\n')
                        # out_file.write('%.6f\n' % cosine[i, j])
                    
    delta = time() - start
    info('finish output file', delta)
