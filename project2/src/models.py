# coding: utf-8

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
import os
import arrow
from bistiming import SimpleTimer
from gensim.models import KeyedVectors
# from keras.layers.embeddings import Embedding
from keras.models import model_from_yaml, Model
from keras.callbacks import Callback

np.random.seed(666)
embedding = KeyedVectors.load('../data/ASBC_embedding.bin')

MAX_REVIEW_LENGTH = 26 # clause 1+2
MAX_SINGLE_REVIEW_LENGTH = 17 # single clause
EMBEDDING_VECTOR_LENGTH = 300
NUM_CLASSES = 4
NUM_UNITS = 100
DROPOUT = 0.2
RECURRENT_DROPOUT = 0.2
NUM_EPOCHS = 100
NUM_BATCHES = 64 # 0 means batch training
VERBOSE = 2
USE_CLASS_WEIGHT = False

DATA_PATH = '../data'
DATA_SUFFIX = '_seg.txt'
TRAIN_PATH = os.path.join(DATA_PATH, 'train' + DATA_SUFFIX)
TEST_PATH = os.path.join(DATA_PATH, 'test' + DATA_SUFFIX)
SAVE_PATH = '../models'
PREDICT_PATH = '../predictions'

int2relation = ['Temporal', 'Contingency', 'Comparison', 'Expansion']
relation2int = {r:i for i, r in enumerate(int2relation)}


def onehot(n, i):
    x = np.zeros(n)
    x[i] = 1
    return x


class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='acc', threshold=0.999, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
    
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('acc')
        if current is None:
            print('Warning: acc is None')
            exit(1)
        
        if current >= self.threshold:
            if self.verbose > 0:
                print('Early stopping: accuracy = %f at epoch %d' % (current, epoch))
            self.model.stop_training = True


class _BaseClass(object):
    def __init__(self):
        pass
    
    def _build_data(self):
        pass
    
    def _build_model(self, verbose=True):
        pass
    
    def fit(self):
        pass
    
    def save(self):
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        yaml = self.model.to_yaml()
        time = arrow.now('Asia/Taipei').format('YYYYMMDD_HH:mm:ss')
        model_filename = '%s_model_%s.yaml' % (self.__class__.__name__, time)
        model_path = os.path.join(SAVE_PATH, model_filename)
        with SimpleTimer('Writing model to %s' % model_path, end_in_new_line=False):
            open(model_path, 'w').write(yaml)
        weights_filename = '%s_weights_%s.h5' % (self.__class__.__name__, time)
        weights_path = os.path.join(SAVE_PATH, weights_filename)
        with SimpleTimer('Writing weights to %s' % weights_path, end_in_new_line=False):
            self.model.save_weights(weights_path)

        return model_path, weights_path
            
    def predict(self, X):
        pass
    

class SimpleRNN(_BaseClass):
    def __init__(self, model_path=None, weights_path=None):
        self.X, self.y, self.class_cnt = self._build_data()

        if model_path and weights_path:
            with open(model_path) as model_file:
                self.model = model_from_yaml(model_file.read())
            self.model.load_weights(weights_path)
        else:
            self.model = self._build_model(verbose=False)

    def _build_data(self):
        with open(TRAIN_PATH) as train_file:
            lines = train_file.readlines()
        
        num_lines = len(lines)
        X = np.zeros([num_lines, MAX_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
        y = np.zeros([num_lines, NUM_CLASSES])
        
        cnt = 0
        class_cnt = [0] * 4
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
                class_cnt[relation2int[relation]] += 1
        
        X, y = X[:cnt], y[:cnt]
        p = np.random.permutation(cnt)
        X, y = X[p], y[p]
        
        global NUM_BATCHES
        if NUM_BATCHES == 0:
            NUM_BATCHES = cnt
        
        return X, y, class_cnt

    def _build_model(self, verbose=True):
        model = Sequential()
        model.add(LSTM(units=NUM_UNITS, input_shape=(MAX_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))
        model.add(Dense(NUM_CLASSES, activation='sigmoid'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if verbose:
            model.summary()
        return model

    def fit(self):
        inv_freq = [1.0/c for c in self.class_cnt]
        sum_ = sum(inv_freq)
        if USE_CLASS_WEIGHT:
            class_weight = {c : w/sum_ for c, w in enumerate(inv_freq)}
        else:
            class_weight = None
        self.model.fit(self.X, self.y, batch_size=NUM_BATCHES, epochs=NUM_EPOCHS, verbose=VERBOSE, class_weight=class_weight, callbacks=[EarlyStoppingByAccuracy()])
    
    def predict(self):
        time = arrow.now('Asia/Taipei').format('YYYYMMDD_HH:mm:ss')
        if not os.path.exists(PREDICT_PATH):
            os.makedirs(PREDICT_PATH)
        pred_path = os.path.join(PREDICT_PATH, 'submission_%s.csv' % time)

        with SimpleTimer('Writing predictions to %s' % pred_path, end_in_new_line=False):
            with open(TEST_PATH) as test_file, open(pred_path, 'w') as output_file:
                output_file.write('Id,Relation\n')
                
                lines = test_file.readlines()
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
                        print('Warning: len(embedi) == 0 for some i = 1, 2.')
                        # exit(0)
                    
                    y_prob = self.model.predict(X)
                    y = np.argmax(y_prob)
                    relation = int2relation[y]
                    
                    output_file.write('%s,%s\n' % (id_, relation))


class ConcatRNN(_BaseClass):
    def __init__(self, model_path=None, weights_path=None):
        self.X1, self.X2, self.y, self.class_cnt = self._build_data()

        if model_path and weights_path:
            with open(model_path) as model_file:
                self.model = model_from_yaml(model_file.read())
            self.model.load_weights(weights_path)
        else:
            self.model = self._build_model(verbose=False)
        
    def _build_data(self):
        with open(TRAIN_PATH) as train_file:
            lines = train_file.readlines()
        
        num_lines = len(lines)
        X1 = np.zeros([num_lines, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
        X2 = np.zeros([num_lines, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
        y = np.zeros([num_lines, NUM_CLASSES])
        
        cnt = 0
        class_cnt = [0] * 4
        for line in lines[1:]:
            id_, clause1, clause2, relation = line.strip().split(',')
            clause1 = clause1.split('|')
            clause2 = clause2.split('|')
            embed1 = [embedding[w] for w in clause1 if w in embedding]
            embed2 = [embedding[w] for w in clause2 if w in embedding]
            embed1.reverse()
            embed2.reverse()
            
            if len(embed1) > 0 and len(embed2) > 0:
                X1[cnt, -len(embed1):] = np.array(embed1)
                X2[cnt, -len(embed2):] = np.array(embed2)
                y[cnt] = onehot(NUM_CLASSES, relation2int[relation])
                cnt += 1
                class_cnt[relation2int[relation]] += 1
        
        X1, X2, y = X1[:cnt], X2[:cnt], y[:cnt]
        p = np.random.permutation(cnt)
        X1, X2, y = X1[p], X2[p], y[p]
        
        global NUM_BATCHES
        if NUM_BATCHES == 0:
            NUM_BATCHES = cnt
        
        return X1, X2, y, class_cnt

    def _build_model(self, verbose=True):
        input1 = Input(shape=(MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dtype='float32')
        input2 = Input(shape=(MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dtype='float32')
        lstm1_out = LSTM(units=NUM_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(input1)
        lstm2_out = LSTM(units=NUM_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(input2)
        concat = keras.layers.concatenate([lstm1_out, lstm2_out])
        output = Dense(NUM_CLASSES, activation='sigmoid')(concat)
        model = Model(inputs=[input1, input2], outputs=[output])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if verbose:
            model.summary()
        return model

    def fit(self):
        inv_freq = [1.0/c for c in self.class_cnt]
        sum_ = sum(inv_freq)
        if USE_CLASS_WEIGHT:
            class_weight = {c : w/sum_ for c, w in enumerate(inv_freq)}
        else:
            class_weight = None
        self.model.fit([self.X1, self.X2], self.y, batch_size=NUM_BATCHES, epochs=NUM_EPOCHS, verbose=VERBOSE, class_weight=class_weight, callbacks=[EarlyStoppingByAccuracy()])
    
    def predict(self):
        time = arrow.now('Asia/Taipei').format('YYYYMMDD_HH:mm:ss')
        if not os.path.exists(PREDICT_PATH):
            os.makedirs(PREDICT_PATH)
        pred_path = os.path.join(PREDICT_PATH, 'submission_%s.csv' % time)

        with SimpleTimer('Writing predictions to %s' % pred_path, end_in_new_line=False):
            with open(TEST_PATH) as test_file, open(pred_path, 'w') as output_file:
                output_file.write('Id,Relation\n')
                
                lines = test_file.readlines()
                for line in lines[1:]:
                    id_, clause1, clause2 = line.strip().split(',')
                    clause1 = clause1.split('|')
                    clause2 = clause2.split('|')
                    embed1 = [embedding[w] for w in clause1 if w in embedding]
                    embed2 = [embedding[w] for w in clause2 if w in embedding]
                    embed1.reverse()
                    embed2.reverse()

                    if len(embed1) > 0 and len(embed2) > 0:
                        X1 = np.zeros([1, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                        X2 = np.zeros([1, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                        X1[0, -len(embed1):] = np.array(embed1)
                        X2[0, -len(embed2):] = np.array(embed2)
                    else:
                        print('Warning: for id = %s, len(embedi) == 0 for some i = 1, 2.' % id_)
                        # exit(0)
                    
                    y_prob = self.model.predict([X1, X2])
                    y = np.argmax(y_prob)
                    relation = int2relation[y]
                    
                    output_file.write('%s,%s\n' % (id_, relation))
