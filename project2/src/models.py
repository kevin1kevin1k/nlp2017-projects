# coding: utf-8

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D
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
DROPOUT = 0.3
RECURRENT_DROPOUT = 0.3
NUM_EPOCHS = 100
NUM_BATCHES = 64 # 0 means batch training
VERBOSE = 2
USE_CLASS_WEIGHT = False
OPTIMIZER = 'rmsprop'
NUM_TOP_WORDS = 10
NUM_FILTER_WORDS = 30

DATA_PATH = '../data'
DATA_SUFFIX = '_seg_CKIP.txt'
SPLIT_SYMBOL = ' '
TRAIN_PATH = os.path.join(DATA_PATH, f'train{DATA_SUFFIX}')
TEST_PATH = os.path.join(DATA_PATH, f'test{DATA_SUFFIX}')
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


class Lang(object):
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'START', 1: 'END', 2: 'UNK'}
        self.relation2word2count = {}
        self.top_words = []
        # self.top_words = [
        #     '並', '但', '再', '卻', '將', '後', '從', '時', '當', '而',
        #     '不過', '之前', '之後', '以後', '但是', '儘管', '因此', '因為', '如果', '如此',
        #     '已經', '所以', '時候', '最後', '為了', '然後', '由於', '終於', '開始', '雖然'
        # ]
        self.topword2index = {}
        self.num_words = 3
        self.NUM_TOP_WORDS = 0
        self.read_files()
        self.count_words()
    
    def read_files(self):
        for filename in [TRAIN_PATH, TEST_PATH]:
            with open(filename) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    
                    clauses = line.strip().split(',')[1:3]
                    for clause in clauses:
                        if filename == TRAIN_PATH:
                            relation = line.strip().split(',')[3]
                            self.add_sentence(clause, relation)
                        else:
                            self.add_sentence(clause)
                    
    def add_sentence(self, sentence, relation=None):
        for word in sentence.split(SPLIT_SYMBOL):
            self.add_word(word)
            
            if relation is not None:
                if relation not in self.relation2word2count:
                    self.relation2word2count[relation] = {}
                for word in sentence.split(SPLIT_SYMBOL):
                    if word not in self.relation2word2count[relation]:
                        self.relation2word2count[relation][word] = 0
                    self.relation2word2count[relation][word] += 1

    def add_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            index = self.num_words
            self.word2index[word] = index
            self.word2count[word] = 1
            self.index2word[index] = word
            self.num_words += 1
    
    def count_words(self):
        count2words = self.w2c_to_c2w(self.word2count)
        counts = list(reversed(sorted(count2words.keys())))
        for c in counts[:NUM_FILTER_WORDS]:
            for word in count2words[c]:
                for relation in self.relation2word2count:
                    if word in self.relation2word2count[relation]:
                        del self.relation2word2count[relation][word]
        
        for relation, word2count in self.relation2word2count.items():
            count2words = self.w2c_to_c2w(word2count)
            counts = list(reversed(sorted(count2words.keys())))
            
            candidates = []
            for c in counts[:NUM_TOP_WORDS]:
                candidates += count2words[c]
            self.top_words += candidates[:NUM_TOP_WORDS]
        
        self.top_words = list(set(self.top_words))
        self.NUM_TOP_WORDS = len(self.top_words)
        self.topword2index = {w:i+1 for i, w in enumerate(self.top_words)}
    
    def w2c_to_c2w(self, w2c):
        c2w = {}
        for w, c in w2c.items():
            if c not in c2w:
                c2w[c] = []
            c2w[c].append(w)
        return c2w
    
    

class _BaseClass(object):
    def __init__(self):
        self.lang = Lang()
    
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
        model_filename = f'{self.__class__.__name__}_model_{time}.yaml'
        model_path = os.path.join(SAVE_PATH, model_filename)
        with SimpleTimer(f'Writing model to {model_path}', end_in_new_line=False):
            open(model_path, 'w').write(yaml)
        weights_filename = f'{self.__class__.__name__}_weights_{time}.h5'
        weights_path = os.path.join(SAVE_PATH, weights_filename)
        with SimpleTimer(f'Writing weights to {weights_path}', end_in_new_line=False):
            self.model.save_weights(weights_path)

        return model_path, weights_path
            
    def predict(self):
        pass
    

class SimpleRNN(_BaseClass):
    def __init__(self, model_path=None, weights_path=None):
        super(SimpleRNN, self).__init__()
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
            clause1 = clause1.split(SPLIT_SYMBOL)
            clause2 = clause2.split(SPLIT_SYMBOL)
            words = clause1 + clause2
            if embed := [embedding[w] for w in words if w in embedding]:
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
        model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
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
        pred_path = os.path.join(PREDICT_PATH, f'submission_{time}.csv')

        with SimpleTimer(f'Writing predictions to {pred_path}', end_in_new_line=False):
            with open(TEST_PATH) as test_file, open(pred_path, 'w') as output_file:
                output_file.write('Id,Relation\n')

                lines = test_file.readlines()
                for line in lines[1:]:
                    id_, clause1, clause2 = line.strip().split(',')
                    clause1 = clause1.split(SPLIT_SYMBOL)
                    clause2 = clause2.split(SPLIT_SYMBOL)
                    words = clause1 + clause2
                    embed = [embedding[w] for w in words if w in embedding]

                    X = np.zeros([1, MAX_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                    if embed:
                        X[0, -len(embed):] = np.array(embed)
                    else:
                        print(f'Warning: for id = {id_}, len(embedi) == 0 for some i = 1, 2.')

                    y_prob = self.model.predict(X)
                    y = np.argmax(y_prob)
                    relation = int2relation[y]

                    output_file.write('%s,%s\n' % (id_, relation))


class ConcatRNN(_BaseClass):
    def __init__(self, model_path=None, weights_path=None):
        super(ConcatRNN, self).__init__()
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
            clause1 = clause1.split(SPLIT_SYMBOL)
            clause2 = clause2.split(SPLIT_SYMBOL)
            embed1 = [embedding[w] for w in clause1 if w in embedding]
            embed2 = [embedding[w] for w in clause2 if w in embedding]
            embed1.reverse()
            embed2.reverse()

            if embed1:
                X1[cnt, MAX_SINGLE_REVIEW_LENGTH-len(embed1):] = np.array(embed1)
            if embed2:
                X2[cnt, MAX_SINGLE_REVIEW_LENGTH-len(embed2):] = np.array(embed2)

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
        model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
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
        pred_path = os.path.join(PREDICT_PATH, f'submission_{time}.csv')

        with SimpleTimer(f'Writing predictions to {pred_path}', end_in_new_line=False):
            with open(TEST_PATH) as test_file, open(pred_path, 'w') as output_file:
                output_file.write('Id,Relation\n')

                lines = test_file.readlines()
                for line in lines[1:]:
                    id_, clause1, clause2 = line.strip().split(',')
                    clause1 = clause1.split(SPLIT_SYMBOL)
                    clause2 = clause2.split(SPLIT_SYMBOL)
                    embed1 = [embedding[w] for w in clause1 if w in embedding]
                    embed2 = [embedding[w] for w in clause2 if w in embedding]
                    embed1.reverse()
                    embed2.reverse()

                    X1 = np.zeros([1, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                    X2 = np.zeros([1, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                    if embed1:
                        X1[0, -len(embed1):] = np.array(embed1)
                    if embed2:
                        X2[0, -len(embed2):] = np.array(embed2)

                    y_prob = self.model.predict([X1, X2])
                    y = np.argmax(y_prob)
                    relation = int2relation[y]

                    output_file.write('%s,%s\n' % (id_, relation))


class ConcatCountRNN(_BaseClass):
    def __init__(self, model_path=None, weights_path=None):
        super(ConcatCountRNN, self).__init__()
        self.X1, self.X2, self.X3, self.y, self.class_cnt = self._build_data()

        if model_path and weights_path:
            with open(model_path) as model_file:
                self.model = model_from_yaml(model_file.read())
            self.model.load_weights(weights_path)
        else:
            self.model = self._build_model(verbose=True)
        
    def _build_data(self):
        with open(TRAIN_PATH) as train_file:
            lines = train_file.readlines()

        num_lines = len(lines)
        X1 = np.zeros([num_lines, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
        X2 = np.zeros([num_lines, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
        X3 = np.zeros([num_lines, 1 + self.lang.NUM_TOP_WORDS*NUM_CLASSES])
        y = np.zeros([num_lines, NUM_CLASSES])

        cnt = 0
        class_cnt = [0] * 4
        for line in lines[1:]:
            id_, clause1, clause2, relation = line.strip().split(',')
            clause1 = clause1.split(SPLIT_SYMBOL)
            clause2 = clause2.split(SPLIT_SYMBOL)
            embed1 = [embedding[w] for w in clause1 if w in embedding]
            embed2 = [embedding[w] for w in clause2 if w in embedding]
            embed1.reverse()
            embed2.reverse()

            if embed1:
                X1[cnt, MAX_SINGLE_REVIEW_LENGTH-len(embed1):] = np.array(embed1)
            if embed2:
                X2[cnt, MAX_SINGLE_REVIEW_LENGTH-len(embed2):] = np.array(embed2)

            for word in clause1 + clause2:
                X3[cnt, self.lang.topword2index.get(word, 0)] += 1

            y[cnt] = onehot(NUM_CLASSES, relation2int[relation])
            cnt += 1
            class_cnt[relation2int[relation]] += 1

        X1, X2, X3, y = X1[:cnt], X2[:cnt], X3[:cnt], y[:cnt]
        p = np.random.permutation(cnt)
        X1, X2, X3, y = X1[p], X2[p], X3[p], y[p]

        global NUM_BATCHES
        if NUM_BATCHES == 0:
            NUM_BATCHES = cnt

        return X1, X2, X3, y, class_cnt

    def _build_model(self, verbose=True):
        input1 = Input(shape=(MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dtype='float32')
        input2 = Input(shape=(MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dtype='float32')
        input3 = Input(shape=(1 + self.lang.NUM_TOP_WORDS*NUM_CLASSES,), dtype='float32')
        lstm1_out = LSTM(units=NUM_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(input1)
        lstm2_out = LSTM(units=NUM_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(input2)
        concat_lstm = keras.layers.concatenate([lstm1_out, lstm2_out])
        concat = keras.layers.concatenate([concat_lstm, input3])
        output = Dense(NUM_CLASSES, activation='sigmoid')(concat)
        model = Model(inputs=[input1, input2, input3], outputs=[output])
        model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
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
        self.model.fit([self.X1, self.X2, self.X3], self.y, batch_size=NUM_BATCHES, epochs=NUM_EPOCHS, verbose=VERBOSE, class_weight=class_weight, callbacks=[EarlyStoppingByAccuracy()])
    
    def predict(self):
        time = arrow.now('Asia/Taipei').format('YYYYMMDD_HH:mm:ss')
        if not os.path.exists(PREDICT_PATH):
            os.makedirs(PREDICT_PATH)
        pred_path = os.path.join(PREDICT_PATH, f'submission_{time}.csv')

        with SimpleTimer(f'Writing predictions to {pred_path}', end_in_new_line=False):
            with open(TEST_PATH) as test_file, open(pred_path, 'w') as output_file:
                output_file.write('Id,Relation\n')

                lines = test_file.readlines()
                for line in lines[1:]:
                    id_, clause1, clause2 = line.strip().split(',')
                    clause1 = clause1.split(SPLIT_SYMBOL)
                    clause2 = clause2.split(SPLIT_SYMBOL)
                    embed1 = [embedding[w] for w in clause1 if w in embedding]
                    embed2 = [embedding[w] for w in clause2 if w in embedding]
                    embed1.reverse()
                    embed2.reverse()

                    X1 = np.zeros([1, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                    X2 = np.zeros([1, MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH])
                    X3 = np.zeros([1, 1 + self.lang.NUM_TOP_WORDS*NUM_CLASSES])

                    if embed1:
                        X1[0, -len(embed1):] = np.array(embed1)
                    if embed2:
                        X2[0, -len(embed2):] = np.array(embed2)

                    for word in clause1 + clause2:
                        X3[0, self.lang.topword2index.get(word, 0)] += 1

                    y_prob = self.model.predict([X1, X2, X3])
                    y = np.argmax(y_prob)
                    relation = int2relation[y]

                    output_file.write('%s,%s\n' % (id_, relation))


class ConvConcatRNN(ConcatRNN):
    def __init__(self, model_path=None, weights_path=None):
        super(ConvConcatRNN, self).__init__()

    def _build_model(self, verbose=True):
        input1 = Input(shape=(MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dtype='float32')
        input2 = Input(shape=(MAX_SINGLE_REVIEW_LENGTH, EMBEDDING_VECTOR_LENGTH), dtype='float32')
        conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input1)
        conv2 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input2)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        lstm1 = LSTM(units=NUM_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(pool1)
        lstm2 = LSTM(units=NUM_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(pool2)
        concat = keras.layers.concatenate([lstm1, lstm2])
        output = Dense(NUM_CLASSES, activation='sigmoid')(concat)
        model = Model(inputs=[input1, input2], outputs=[output])
        model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
        if verbose:
            model.summary()
        return model
