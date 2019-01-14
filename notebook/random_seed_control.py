# https://www.tensorflow.org/guide/performance/overview
# https://software.intel.com/en-us/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus
# Not all options are necessary to control randomness
"""
Different behaviors during training and testing

Some Keras layers (e.g. Dropout, BatchNormalization) behave differently at training time and testing time.
You can tell whether a layer uses the "learning phase" (train/test) by printing layer.uses_learning_phase,
a boolean: True if the layer has a different behavior in training mode and test mode, False otherwise.

If your model includes such layers, then you need to specify the value of the learning phase as part of feed_dict,
so that your model knows whether to apply dropout/etc or not.

To make use of the learning phase, simply pass the value "1" (training mode) or "0" (test mode) to feed_dict:
"""

NUM_PARALLEL_EXEC_UNITS = 1

import keras
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,   
                              inter_op_parallelism_threads=1,    # this one option can control randomness. But why loss 0.2 -> 4? update tensorflow from 1.8 to 1.14 solves the problem
                              # following lines don't matter
                              allow_soft_placement=True, 
                              device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS}
                              )
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#sess = tf.Session()
from keras.layers import Dropout, Dense, LSTM
from keras import backend as K
K.set_session(sess)

import os
os.environ['PYTHONHASHSEED'] = '42'

os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)

os.environ["KMP_BLOCKTIME"] = "30"

os.environ["KMP_SETTINGS"] = "1"

os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# load data
nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
                           
model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probaility distribution, that is
                                 # that its values are all non-negative and sum to 1.
                                 
                                 
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(X_train, Y_train,
          batch_size=128, epochs=10,
          verbose=1,
          validation_data=(X_test, Y_test))