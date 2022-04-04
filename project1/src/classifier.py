#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from gensim.models import KeyedVectors
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from embed import *
from time import time

from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier


def onehot(aspect):
    x = np.zeros(5)
    x[aspect2idx[aspect]] = 1
    return x


def train_data():
    start = time()
    with open('../data/polarity_review_aspect.txt') as in_file:
        line = in_file.readline()
        l = 0

        # L = number of instances.
        # A review with 3 aspects counts as 3 instances.
        L = 165827*5
        # L = 10000*5
        X = np.zeros([L, 300+5])
        # X = np.random.random([L, 300+5])
        y = np.zeros(L)
        while line != '':
            review = line.strip()
            words = review.split('|')
            vec = sent2vec(words)

            line = in_file.readline()
            pos = line.strip().split()

            line = in_file.readline()
            neg = line.strip().split()

            line = in_file.readline()
            cosine = float(line.strip())

            for aspect in aspect2idx:
                X[l] = np.append(vec, onehot(aspect))
                if aspect in pos:
                    y[l] = +1
                elif aspect in neg:
                    y[l] = -1
                else:
                    y[l] = 0
                l += 1

            # if l + 1 == L:
            #     break

            line = in_file.readline()

        print(X.shape, y.shape)

    delta = time() - start
    info('finish creating X and y', delta)

    return X, y
    

def valid_data():
    start = time()
    with open('../data/aspect_review_seg.txt') as in_file:
        line = in_file.readline()
        l = 0
        
        # L = number of instances.
        # A review with 3 aspects counts as 3 instances.
        L = 200*5
        X_valid = np.zeros([L, 300+5])
        # X_valid = np.random.random([L, 300+5])
        y_valid = np.zeros(L)
        while line != '':
            id_ = int(line.strip())
            
            line = in_file.readline()
            review = line.strip()
            words = review.split('|')
            vec = sent2vec(words)

            line = in_file.readline()
            pos = line.strip().split()

            line = in_file.readline()
            neg = line.strip().split()

            for aspect in aspect2idx:
                X_valid[l] = np.append(vec, onehot(aspect))
                if aspect in pos:
                    y_valid[l] = +1
                elif aspect in neg:
                    y_valid[l] = -1
                else:
                    y_valid[l] = 0
                l += 1
            
            line = in_file.readline()

        print(X_valid.shape, y_valid.shape)
        
    delta = time() - start
    info('finish creating valid', delta)
    
    return X_valid, y_valid


# some commented are too slow
models = [
    ExtraTreesClassifier,
    KNeighborsClassifier,
    # GaussianNB,
    LinearSVC,
    # SVC,
    DecisionTreeClassifier,
    RandomForestClassifier,
    # AdaBoostClassifier,
    GradientBoostingClassifier,
    # SGDClassifier,
]

names = [
    'ExtraTreesClassifier',
    'KNeighborsClassifier',
    # 'GaussianNBClassifier',
    'LinearSVC',
    # 'SVC',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    # 'AdaBoostClassifier',
    'GradientBoostingClassifier',
    # 'SGDClassifier',
]


def run_exp(model, name, X, y, X_valid, y_valid, d):
    start = time()
    clf = model(max_depth=d)
    clf.fit(X, y)
    delta = time() - start
    info(f'finish fitting {name}', delta)
    print(d)

    y_pred = clf.predict(X_valid)
    print(accuracy_score(y_valid, y_pred))

    joblib.dump(clf, '%s_%d.pkl' % (name, d))
    print('')


if __name__ == '__main__':
    X, y = train_data()
    X_valid, y_valid = valid_data()
    
    print('')
    # for model, name in zip(models, names):
    #     run_exp(model, name, X, y, X_valid, y_valid)
        # run_exp(model, name, np.concatenate([X, X_valid]), np.concatenate([y, y_valid]), X_valid, y_valid)

    for d in [4, 8, 16, 32, 64]:
        run_exp(RandomForestClassifier, 'RandomForestClassifier', X, y, X_valid, y_valid, d)
        

    # run_exp(GradientBoostingClassifier, 'GradientBoostingClassifier', X, y, X_valid, y_valid)
    # run_exp(GradientBoostingClassifier, 'GradientBoostingClassifier', X, y, X_valid, y_valid)
    # run_exp(SGDClassifier, 'SGDClassifier', X, y, X_valid, y_valid)
    
