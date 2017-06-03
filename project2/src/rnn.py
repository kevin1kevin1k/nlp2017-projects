# coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_yaml
import tensorflow as tf
import os
import arrow
import gc
from bistiming import SimpleTimer
import argparse
import sys
from gensim.models import KeyedVectors

# set up GPU usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_session(tf.Session(config=config))

# constants and settings
np.random.seed(666)
embedding = KeyedVectors.load('../data/ASBC_embedding.bin')
MAX_REVIEW_LENGTH = 26
EMBEDDING_VECTOR_LENGTH = 300
NUM_CLASSES = 4
NUM_UNITS = 100
DROPOUT = 0.2
RECURRENT_DROPOUT = 0.2
NUM_EPOCHS = 3
NUM_BATCHES = 64
VERBOSE = 2
SAVE_PATH = '../models'
PREDICT_PATH = '../predictions'

int2relation = ['Temporal', 'Contingency', 'Comparison', 'Expansion']
relation2int = {r:i for i, r in enumerate(int2relation)}


def onehot(n, i):
    x = np.zeros(n)
    x[i] = 1
    return x


def build_model(verbose=True):
    model = Sequential()
    # TODO: Bidirectional
    model.add(LSTM(units=NUM_UNITS, input_shape = (MAX_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    # TODO: RMSprop
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        model.summary()
    return model


def train():
    with SimpleTimer('Build data', end_in_new_line=False):
        with open('../data/train_seg.txt') as train_file:
            lines = train_file.readlines()
        
        num_lines = len(lines)
        X = np.zeros([num_lines, MAX_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
        y = np.zeros([num_lines, NUM_CLASSES])
        
        cnt = 0
        for line in lines[1:]:
            id_, clause1, clause2, relation = line.strip().split(',')
            clause1 = clause1.split('|')
            clause2 = clause2.split('|')
            words = clause1 + clause2
            embed = [embedding[w] for w in words if w in embedding]
            if len(embed) > 0:
                X[cnt, -len(embed):] = np.array(embed)
                y[cnt] = onehot(NUM_CLASSES, relation2int[relation])
                cnt += 1
        
        X, y = X[:cnt], y[:cnt]
        p = np.random.permutation(cnt)
        X, y = X[p], y[p]

    with SimpleTimer('Train', end_in_new_line=True):
        model = build_model(verbose=True)
        model.fit(X, y, batch_size=NUM_BATCHES, epochs=NUM_EPOCHS, verbose=VERBOSE)

    with SimpleTimer('Save model', end_in_new_line=True):
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        yaml = model.to_yaml()
        time = arrow.now('Asia/Taipei').format('YYYYMMDD_HH:mm:ss')
        yaml_filename = os.path.join(SAVE_PATH, 'rnn_model_%s.yaml' % time)
        with SimpleTimer('Writing model to %s' % yaml_filename, end_in_new_line=False):
            open(yaml_filename, 'w').write(yaml)
        h5py_filename = os.path.join(SAVE_PATH, 'rnn_weights_%s.h5' % time)
        with SimpleTimer('Writing weights to %s' % h5py_filename, end_in_new_line=False):
            model.save_weights(h5py_filename)
    
    return yaml_filename, h5py_filename
        

def test(model_filename, weights_filename):
    with SimpleTimer('Load model', end_in_new_line=False):
        with open(model_filename) as model_file:
            model = model_from_yaml(model_file.read())
        model.load_weights(weights_filename)

    with SimpleTimer('Test', end_in_new_line=True):
        with open('../data/test_seg.txt') as test_file:
            lines = test_file.readlines()
        num_lines = len(lines)
        
        time = arrow.now('Asia/Taipei').format('YYYYMMDD_HH:mm:ss')
        if not os.path.exists(PREDICT_PATH):
            os.makedirs(PREDICT_PATH)
        pred_path = os.path.join(PREDICT_PATH, 'submission_%s.csv' % time)
        with SimpleTimer('Writing predictions to %s' % pred_path, end_in_new_line=False):
            with open(pred_path, 'w') as output_file:
                output_file.write('Id,Relation\n')
                
                for line in lines[1:]:
                    id_, clause1, clause2 = line.strip().split(',')
                    clause1 = clause1.split('|')
                    clause2 = clause2.split('|')
                    words = clause1 + clause2
                    embed = [embedding[w] for w in words if w in embedding]
                    if len(embed) > 0:
                        X = np.zeros([1, MAX_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                        X[0, -len(embed):] = np.array(embed)
                    else:
                        print('len(embed) == 0. Exiting')
                        exit(0)
                    
                    y = model.predict(X)
                    y = np.argmax(y)
                    y = int2relation[y]
                    
                    output_file.write('%s,%s\n' % (id_, y))
                

def get_config():
    parser = argparse.ArgumentParser(description='rnn model')
    parser.add_argument('action', nargs='+')
    parser.add_argument('-m', '--model-filename')
    parser.add_argument('-w', '--weights-filename')
    parser.add_argument('-e', '--epochs')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = get_config()
    if config.epochs:
        NUM_EPOCHS = int(config.epochs)
    if 'train' in config.action:
        model_filename, weights_filename = train()
    if 'test' in config.action:
        if 'train' in config.action:
            pass
        elif config.model_filename and config.weights_filename:
            model_filename, weights_filename = config.model_filename, config.weights_filename
        else:
            print('No model file specified for testing. Exiting.')
            exit()
        
        test(model_filename, weights_filename)
    
    gc.collect()
