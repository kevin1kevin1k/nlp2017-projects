# coding: utf-8

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import os
import gc
from bistiming import SimpleTimer
import argparse
from models import SimpleRNN, ConcatRNN

# set up GPU usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_session(tf.Session(config=config))

MODEL = ConcatRNN

def train():
    with SimpleTimer('Build data and model', end_in_new_line=False):
        model = MODEL()
    with SimpleTimer('Train', end_in_new_line=True):
        model.fit()
    model_path, weights_path = model.save()
    return model_path, weights_path
        

def test(model_path, weights_path):
    with SimpleTimer('Test', end_in_new_line=True):
        with SimpleTimer('Load model', end_in_new_line=False):
            model = MODEL(model_path, weights_path)
        model.predict()
                

def get_config():
    parser = argparse.ArgumentParser(description='rnn model')
    parser.add_argument('action', nargs='+')
    parser.add_argument('-m', '--model-path')
    parser.add_argument('-w', '--weights-path')
    parser.add_argument('-e', '--epochs')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = get_config()
    if config.epochs:
        NUM_EPOCHS = int(config.epochs)
    if 'train' in config.action:
        model_path, weights_path = train()
    if 'test' in config.action:
        if 'train' in config.action:
            pass
        elif config.model_path and config.weights_path:
            model_path, weights_path = config.model_path, config.weights_path
        else:
            print('No model file specified for testing. Exiting.')
            exit()
        
        test(model_path, weights_path)
    
    gc.collect()
