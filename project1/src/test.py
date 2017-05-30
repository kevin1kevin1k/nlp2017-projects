#!/usr/bin/env python2
# coding: utf-8

# from seg import seg
from sklearn.externals import joblib
from classifier import onehot
from embed import sent2vec
import numpy as np


reviews = {}

# always return 0 regardless of review_id and aspect
def all_zero(review_id, aspect):
    return 0


def use_classifier(review_id, aspect):
    clf = joblib.load('RandomForestClassifier.pkl')
    review = reviews[review_id]
    vec = sent2vec(review)
    X = np.append(vec, onehot(aspect))
    X = X.reshape(1, -1)
    pred = clf.predict(X)
    # print(pred)
    return int(pred[0])

def read_test_review():
    with open('../data/test_review_seg.txt') as test_review_file:
        lines = test_review_file.readlines()
    for i in range(len(lines) // 2):
        review_id, review = int(lines[2*i].strip()), lines[2*i+1].strip()
        reviews[review_id] = review
    
    return reviews


def make_prediction(get_label):
    import pandas as pd
    
    reviews = read_test_review()
    df_test = pd.read_csv('../data/test.csv')
    df_label = df_test[['Review_id', 'Aspect']].apply(lambda x: get_label(*x), axis=1).to_frame()
    df_concat = pd.concat([df_test[['Id']], df_label], axis=1)
    df_concat.columns = ('Id', 'Label')
    df_concat.to_csv('submission.csv', index=False)
    
    return df_concat


make_prediction(use_classifier)

