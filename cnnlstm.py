# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:03:26 2019

@author: Himanshu
"""


from keras.models import Sequential
from keras.layers import Flatten, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import MaxPooling3D
from keras.layers import Dense
from keras.layers import Dropout


# Initialising
def cnnlstm(frames, channels, pixels_x, pixels_y):
    classifier = Sequential()
    classifier.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),
                       input_shape=(frames, channels, pixels_x, pixels_y),
                       padding='same', return_sequences=True))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_first'))
    
    classifier.add(ConvLSTM2D(filters=128, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_first'))
    
    
    
    classifier.add(Flatten())
        
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    '''
    classifier.add(BatchNormalization())
    classifier.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format = 'channels_first'))'''
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



