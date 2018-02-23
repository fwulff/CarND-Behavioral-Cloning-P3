#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:17:19 2017

@author: florian
"""
#import processing and generating functions
import generator

# Import all libraries
import os
import json
import errno
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, ELU, Lambda, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np

def balance_dataset(delete_prob = 0., delete_outliers = False):
    # read original file
    data = pd.read_csv('./data/data/driving_log.csv')
        
    # create histogram of steering angles
    # histogram = data.steering.hist(bins=25)
    # histogram.plot()
    print(data.describe())
    
    # Percentage between 0 and 100
    # delete_prob = 25
    
    index_array = []
    # delete steering angles of 0. to reduce bias towards straight driving
    
    for index in range(len(data)):
        if (data.steering[index] == 0.):
            rand_prob = np.random.randint(0,100)
            if rand_prob < delete_prob:
                index_array.append(index)
    
    data = data.drop(index_array)
    
    # delete all entries with 0. angle
    # data = data[data.steering != 0.]
    
    print("Number of entries deleted: ", len(index_array))
    
    if delete_outliers == True:
        # delete steering angles of >0.5 or <-0.5 to reduce oversteering
        data = data[data.steering < 0.5]
        data = data[data.steering > -0.5]
    
    # create histogram of steering angles
    histogram_new = data.steering.hist(bins=25)
    histogram_new.plot()
    print(data.describe())
    
    #write to new file
    data.to_csv('./data/data/driving_log_new.csv')
    
    return(len(data))
    
# using non-resized udacity images
# ch, row, col = 3, 160, 320  # camera format

# using smallest resized udacity images
ch, row, col = 3, 60, 80  # camera format for NVIDIA first and CommaAI
# ch, row, col = 3, 66, 200


# balance dataset
nb_of_data_entries = balance_dataset(delete_prob = 25, delete_outliers = False)

# Train model for n epochs and a batch size of x
epochs = 3
batch_size = 64
nb_train_samples = 20048
nb_validation_samples = 2048

# nb_train_samples = ((nb_of_data_entries - batch_size) * 3)
# nb_validation_samples = np.rint(nb_train_samples)

# learning_rate = 0.001
learning_rate = 0.0001

# Define model using sequential model

# Use commo.ai model or NVIDIA model?
CommaModel = False

if CommaModel == True:
    model = Sequential()
    #model.add(Lambda(lambda x: x/127.5 - 1.,
    #            input_shape=(row, col, ch),
    #            output_shape=(row, col, ch)))
    model.add(BatchNormalization(axis=1, input_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

# NVIDIA model
else:
    """
    model = Sequential()
    model.add(BatchNormalization(axis=1, input_shape=(row, col, ch)))
    model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'))
    model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    """
    
    model = Sequential()
    model.add(BatchNormalization(axis=1, input_shape=(row, col, ch)))
    model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'))
    model.add(Convolution2D(48, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    """
    model = Sequential()
    model.add(BatchNormalization(axis=1, input_shape=(row, col, ch)))        
    model.add(Convolution2D(24, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    #model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    #model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    #model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    #model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    #model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    """
    
model.summary()

#create custom adam optimizer with lower training rate
model.compile(loss='mse', optimizer=Adam(learning_rate))
#model.compile(optimizer="adam", loss="mse")

# Delete old training files if existent
# http://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist

try:
    os.remove('model.h5')

except OSError as error:
    if error.errno != errno.ENOENT:
        raise

try:
    os.remove('model.json')

except OSError as error:
    if error.errno != errno.ENOENT:
        raise

# Keras image preprocessing https://keras.io/preprocessing/image/
# Not applicable, because angles are directly related to rotation, shear etc and would be wrong after augmentation
"""
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=15,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range = 200.,
    zoom_range = 0.,
    vertical_flip = False,
    horizontal_flip= False,
    rescale = None)
"""

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(X_train)

json_string = model.to_json()
with open('model.json', 'w') as file:
    json.dump(json_string, file)

print("model saved as model.json")

train_data_gen = generator.generate_train_batch(batch_size)

val_data_gen = generator.generate_validation_batch(batch_size)

# create checkpoints for each improvement
# use MSE as val_loss
checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')

# early termination with epochs as patience = 2 to prevent overfitting
callback = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

# fits the model on batches with real-time data augmentation:
# https://keras.io/models/model/
history = model.fit_generator(train_data_gen,
                    samples_per_epoch=nb_train_samples, 
                    nb_epoch=epochs, 
                    verbose=1, 
                    callbacks=[checkpoint, callback], 
                    validation_data=val_data_gen,
                    nb_val_samples = nb_validation_samples)

print("Training completed")



model.save_weights('model.h5')
print("weights stored as model.h5")
