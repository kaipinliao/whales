# https://www.hhllcks.de/blog/2018/5/4/version-your-machine-learning-models-with-sacred
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import os
import numpy as np
import random as rn
from tqdm import tqdm
import matplotlib.pyplot as plt

target_height = 32
target_width  = 32
target_channel = 1
epochs = 20
batch_size = 32

augment = [{
#        'rotation_range': 40,
    'rescale': 1./255,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
}]

convolution_layers = [
    {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'},
    {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'},
#        {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'},
    {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'}
]
maxpooling_pool_size = (2, 2)
maxpooling_dropout = 0.0

dense_layers = [
    {'size': 64, 'activation': 'relu'}
#        {'size': 64, 'activation': 'relu'}
]

dense_dropout = 0.0
final_dropout = 0.5

data_version_number = '0_1'

message = 'still testing reproducible. remove dropout. Optimizer: rmsprop'

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.normalization import BatchNormalization

#############################################################################
# https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# Apparently you may use different seed values at each stage
seed_value= 0
NUM_PARALLEL_EXEC_UNITS = 1

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                              inter_op_parallelism_threads=1,
                              allow_soft_placement=True, 
                              device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS}
                              )

from keras import backend as K

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)

os.environ["KMP_BLOCKTIME"] = "30"

os.environ["KMP_SETTINGS"] = "1"

os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
#############################################################################
        
TRAIN_PATH = 'C:/Users/KaiPin Liao/Documents/kaggle_whales/data/train_' + data_version_number + '/'
VALIDATION_PATH = 'C:/Users/KaiPin Liao/Documents/kaggle_whales/data/validation_' + data_version_number + '/'
TEST_PATH  = 'C:/Users/KaiPin Liao/Documents/kaggle_whales/data/test_' + data_version_number + '/'

####################################################################################

model = Sequential()
####################################################################################
#    model.add(Conv2D(32, (3, 3), input_shape=(target_height, target_width, target_channel)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    
#    model.add(Conv2D(32, (3, 3)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    
#    model.add(Conv2D(64, (3, 3)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
####################################################################################

# VGG-like
#model.add(Conv2D(convolution_layers[0]['filters'],
#                 kernel_size = convolution_layers[0]['kernel_size'],
#                 activation  = convolution_layers[0]['activation'],
#                 input_shape = (target_height, target_width, target_channel)))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size = maxpooling_pool_size))
#
#for layer in convolution_layers[1:]:
#    model.add(Conv2D(layer['filters'],
#                     kernel_size = layer['kernel_size'],
#                     activation  = layer['activation']))
#    model.add(BatchNormalization())
#    model.add(MaxPooling2D(pool_size = maxpooling_pool_size))
#    model.add(Dropout(maxpooling_dropout))
# the model so far outputs 3D feature maps (height, width, features)
model.add(Dense(32, input_shape=(target_height, target_width, target_channel)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#    model.add(Dense(64))
#    model.add(Activation('relu'))

for layer in dense_layers:
    model.add(Dense(layer['size'], activation = layer['activation']))
#    model.add(BatchNormalization())
    if layer != dense_layers[-1]:
        model.add(Dropout(dense_dropout))
        
model.add(Dropout(final_dropout))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss      = 'categorical_crossentropy',
              optimizer = 'rmsprop', #'Adadelta',
              metrics   = ['accuracy'])

#print(model.summary())

#    train_datagen = ImageDataGenerator(
#                        rescale = augment[0]['rescale'],
##                        rotation_range = augment[0]['rotation_range'],
#                        shear_range = augment[0]['shear_range'],
#                        zoom_range = augment[0]['zoom_range'],
#                        horizontal_flip = augment[0]['horizontal_flip']
#                    )
train_datagen = ImageDataGenerator(
                    rescale = augment[0]['rescale']
                )
    
validation_datagen = ImageDataGenerator(rescale = augment[0]['rescale'])

history = keras.callbacks.History()

train_generator = train_datagen.flow_from_directory(TRAIN_PATH, 
                                                    target_size = (target_height, target_width),
                                                    batch_size = batch_size,
                                                    color_mode = 'grayscale',
                                                    shuffle = False
                                                   )

validation_generator = validation_datagen.flow_from_directory(VALIDATION_PATH, 
                                                    target_size = (target_height, target_width),
                                                    batch_size = batch_size,
                                                    color_mode = 'grayscale',
                                                    shuffle = False
                                                   )

history = model.fit_generator(train_generator, 
                              validation_data = validation_generator,
                              epochs = epochs,
                              verbose = 2,
                              steps_per_epoch = 342 // batch_size,
                              validation_steps= 114 // batch_size,
                              callbacks = [
                                      ModelCheckpoint("weights.hdf5", monitor='val_loss',
                                                      save_best_only=True, mode='auto', period=1, verbose=0)
                              ]
                             )

model.load_weights("weights.hdf5")
model_loss, model_acc = model.evaluate_generator(generator = train_generator,
                                                 steps = 342 // batch_size)                              
model_val_loss, model_val_acc = model.evaluate_generator(generator = validation_generator,
                                                         steps = 114 // batch_size)

print('best model metrics on train set: ', str(round(model_acc, 4)), str(round(model_loss, 4)))
print('best model metrics on valid set: ', str(round(model_val_acc, 4)), str(round(model_val_loss, 4)))
print(str(round(model_val_acc, 4)), str(round(model_val_loss, 4)),
        str(round(model_acc, 4)), str(round(model_loss, 4)))