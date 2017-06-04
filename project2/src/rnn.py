# coding: utf-8

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_yaml, Model
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
MAX_REVIEW_LENGTH = 26 # clause 1+2
MAX_SINGLE_REVIEW_LENGTH = 17 # single clause
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
DATA_PATH = '../data'
DATA_SUFFIX = '_seg.txt'
TRAIN_PATH = os.path.join(DATA_PATH, 'train' + DATA_SUFFIX)
TEST_PATH = os.path.join(DATA_PATH, 'test' + DATA_SUFFIX)

int2relation = ['Temporal', 'Contingency', 'Comparison', 'Expansion']
relation2int = {r:i for i, r in enumerate(int2relation)}


def onehot(n, i):
    x = np.zeros(n)
    x[i] = 1
    return x


def build_model(verbose=True):
    model = Sequential()
    model.add(LSTM(units=NUM_UNITS, input_shape=(MAX_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        model.summary()
    return model


def build_concat_model(verbose=True):
    input1 = Input(shape=(MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dtype='float32')
    input2 = Input(shape=(MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dtype='float32')
    lstm1_out = LSTM(units=NUM_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(input1)
    lstm2_out = LSTM(units=NUM_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(input2)
    concat = keras.layers.concatenate([lstm1_out, lstm2_out])
    output = Dense(NUM_CLASSES, activation='sigmoid')(concat)
    model = Model(inputs=[input1, input2], outputs=[output])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        model.summary()
    return model


def build_data():
    with open(TRAIN_PATH) as train_file:
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
    
    return X, y


def build_data_for_concat_model():
    with open(TRAIN_PATH) as train_file:
        lines = train_file.readlines()
    
    num_lines = len(lines)
    X1 = np.zeros([num_lines, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
    X2 = np.zeros([num_lines, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
    y = np.zeros([num_lines, NUM_CLASSES])
    
    cnt = 0
    for line in lines[1:]:
        id_, clause1, clause2, relation = line.strip().split(',')
        clause1 = clause1.split('|')
        clause2 = clause2.split('|')
        embed1 = [embedding[w] for w in clause1 if w in embedding]
        embed2 = [embedding[w] for w in clause2 if w in embedding]
        
        if len(embed1) > 0 and len(embed2) > 0:
            X1[cnt, -len(embed1):] = np.array(embed1)
            X2[cnt, -len(embed2):] = np.array(embed2)
            y[cnt] = onehot(NUM_CLASSES, relation2int[relation])
            cnt += 1
    
    X1, X2, y = X1[:cnt], X2[:cnt], y[:cnt]
    p = np.random.permutation(cnt)
    X1, X2, y = X1[p], X2[p], y[p]
    
    return X1, X2, y


def train():
    with SimpleTimer('Build data', end_in_new_line=False):
        # X, y = build_data()
        X1, X2, y = build_data_for_concat_model()
        
    with SimpleTimer('Train', end_in_new_line=True):
        # model = build_model(verbose=True)
        model = build_concat_model(verbose=True)
        # model.fit(X, y, batch_size=NUM_BATCHES, epochs=NUM_EPOCHS, verbose=VERBOSE)
        model.fit([X1, X2], y, batch_size=NUM_BATCHES, epochs=NUM_EPOCHS, verbose=VERBOSE)

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
    model_path = os.path.join(SAVE_PATH, model_filename)
    weights_path = os.path.join(SAVE_PATH, weights_filename)
    with SimpleTimer('Load model', end_in_new_line=False):
        with open(model_path) as model_file:
            model = model_from_yaml(model_file.read())
        model.load_weights(weights_path)

    with SimpleTimer('Test', end_in_new_line=True):
        with open(TEST_PATH) as test_file:
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
                    # words = clause1 + clause2
                    # embed = [embedding[w] for w in words if w in embedding]
                    embed1 = [embedding[w] for w in clause1 if w in embedding]
                    embed2 = [embedding[w] for w in clause2 if w in embedding]
                    # if len(embed) > 0:
                    if len(embed1) > 0 and len(embed2) > 0:
                        # X = np.zeros([1, MAX_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                        # X[0, -len(embed):] = np.array(embed)
                        X1 = np.zeros([1, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                        X2 = np.zeros([1, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                        X1[0, -len(embed1):] = np.array(embed1)
                        X2[0, -len(embed2):] = np.array(embed2)
                    else:
                        print('Warning: len(embedi) == 0 for some i = 1, 2.')
                        # exit(0)
                    
                    # y = model.predict(X)
                    y = model.predict([X1, X2])
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
