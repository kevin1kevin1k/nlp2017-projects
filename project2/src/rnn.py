
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

# set up GPU usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_session(tf.Session(config=config))

# constants and settings
np.random.seed(666)
TOP_WORDS = 5000
MAX_REVIEW_LENGTH = 500
EMBEDDING_VECTOR_LENGTH = 32
NUM_UNITS = 100
DROPOUT = 0.2
RECURRENT_DROPOUT = 0.2
NUM_EPOCHS = 3
NUM_BATCHES = 64
SAVE_PATH = 'models'


def build_model(verbose=True):
    model = Sequential()
    model.add(Embedding(input_dim=TOP_WORDS, output_dim=EMBEDDING_VECTOR_LENGTH, input_length=MAX_REVIEW_LENGTH))
    model.add(LSTM(units=NUM_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if verbose:
        model.summary()
    return model


def train():
    print(NUM_EPOCHS)
    sys.stdout.flush()
    
    with SimpleTimer('Build data', end_in_new_line=True):
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=TOP_WORDS)
        X_train = sequence.pad_sequences(X_train, maxlen=MAX_REVIEW_LENGTH)
        X_test = sequence.pad_sequences(X_test, maxlen=MAX_REVIEW_LENGTH)

    
    with SimpleTimer('Train', end_in_new_line=True):
        model = build_model(verbose=True)
        model.fit(X_train, y_train, batch_size=NUM_BATCHES, epochs=NUM_EPOCHS, verbose=1, validation_data=(X_test, y_test))
        loss, acc = model.evaluate(X_test, y_test)
    print('Accuracy after %d epochs: %.2f' % (NUM_EPOCHS, acc))
    sys.stdout.flush()

    with SimpleTimer('Saving models', end_in_new_line=True):
        # save model and model weights
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
    
    return 
        

def test():
    pass
    # with SimpleTimer('Testing'):
        
        # model = open('')


def get_config():
    parser = argparse.ArgumentParser(description='rnn model')
    parser.add_argument('action', nargs='+')
    parser.add_argument('-m', '--model-file')
    parser.add_argument('-w', '--weight-file')
    parser.add_argument('-e', '--epochs')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = get_config()
    if config.epochs:
        NUM_EPOCHS = int(config.epochs)
    for action in config.action:
        eval(action)()
    
    gc.collect()
