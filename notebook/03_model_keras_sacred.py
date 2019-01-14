#https://www.hhllcks.de/blog/2018/5/4/version-your-machine-learning-models-with-sacred

import keras
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment('hello_config_keras')
ex.observers.append(MongoObserver.create(url='localhost:27017',
                                         db_name='hello_config_keras2'))
#ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def my_config():
    target_height = 32
    target_width  = 32
    target_channel = 3
    epochs = 10
    
    final_dropout = 0.5
    
    data_version_number = '0_1'


@ex.capture
def log_performance(_run, logs):
#    _run.add_artifact("weights.hdf5")
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("accuracy", float(logs.get('acc')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("val_accuracy", float(logs.get('val_acc')))
    _run.result = (str(round(logs.get('val_acc'), 4)), str(round(logs.get('val_loss'), 4)))

# main script that will run automatically
@ex.automain
def my_main(target_height, target_width, target_channel, epochs, 
            data_version_number):
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras.callbacks import ModelCheckpoint, Callback

    class LogPerformance(Callback):
        def on_epoch_end(self, _, logs={}):
            log_performance(logs=logs)
            
    TRAIN_PATH = 'C:/Users/KaiPin Liao/Documents/kaggle_whales/data/train_' + data_version_number + '/'
    VALIDATION_PATH = 'C:/Users/KaiPin Liao/Documents/kaggle_whales/data/validation_' + data_version_number + '/'
    TEST_PATH  = 'C:/Users/KaiPin Liao/Documents/kaggle_whales/data/test_' + data_version_number + '/'
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(target_height, target_width, target_channel)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # the model so far outputs 3D feature maps (height, width, features)
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(final_dropout))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    print(model.summary())
    
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    
    history = keras.callbacks.History()
    
    train_generator = train_datagen.flow_from_directory(TRAIN_PATH, 
                                                        target_size = (target_height, target_width)
    #                                                     color_mode = 'grayscale'
                                                       )
    
    validation_generator = test_datagen.flow_from_directory(VALIDATION_PATH, 
                                                        target_size = (target_height, target_width)
    #                                                     color_mode = 'grayscale'
                                                       )
    
    history = model.fit_generator(train_generator, 
                                  validation_data = validation_generator,
                                  epochs = epochs,
#                                  callbacks=[history]
                                  callbacks = [LogPerformance()]
                                 )
    
    print('final metrics on train set: ', str(round(history.history['acc'][-1], 4)), str(round(history.history['loss'][-1], 4)))
    print('final metrics on valid set: ', str(round(history.history['val_acc'][-1], 4)), str(round(history.history['val_loss'][-1], 4)))
    return (str(round(history.history['val_acc'][-1], 4)), str(round(history.history['val_loss'][-1], 4)))
    
    
    # list all data in history
#    print(history.history.keys())
#    
#    # summarize history for accuracy
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    # plt.legend(['train'], loc='upper left')
#    plt.show()
#    
#    # summarize history for loss
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    # plt.legend(['train'], loc='upper left')
#    plt.show()