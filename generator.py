#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:44:00 2017

@author: florian
"""
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import adjust_gamma
from scipy.misc import imresize

# Preprocessing and generator for datasets
# ch, row, col = 3, 32, 64  # camera format
ch, row, col = 3, 80, 80  # camera format

# ch, row, col = 3, 150, 200  # camera format after resize
#ch, row, col = 3, 100, 200  # camera format after resize

# crop to reach 200x66

# crop_top = 50
# crop_bottom = 34

crop_top = 20
crop_bottom = 0
    
def process_image(image):
    
    #use YUV 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    # use Y only (luminosity)
    # image = image[:, :, 0]

    # resize to row, col, ch
    image = imresize(image, (row, col, ch)).astype(np.float32)

    #adjust gamma
    image = adjust_gamma(image)

    #crop 
    image = image[crop_top:row-crop_bottom, :, :].astype(np.float32)

    return image
    
def augment_image_angle_data(image, angle, flip_prob = 0.5):
    
    #random flip
    flip_rand = np.random.randint(0, 100)
    if flip_rand >= np.rint((flip_prob*100)):
        image = np.fliplr(image)
        angle = -1 * angle
    
    return image, angle
    #random rotation?
    
    #random shear?

def generate_train_images(batch_size = 64, prob_center = 0.4, prob_left = 0.3, prob_right = 0.3):
    
    data = pd.read_csv('./data/data/driving_log_new.csv')
    random_images_indexes = np.random.randint(0, len(data), batch_size)
    images_angles = []
    
    #probabilities for choosing center, right or left images (sum to 1)
    #as dataset is strongly biased towards straight driving
    prob_center = 0.2
    prob_left = 0.4
    prob_right = 0.4
    
    prob_position = np.random.choice(3, 1, p=[prob_center, prob_right, prob_left])
       
    #adjust angle with regard to camera position
    # offset=1.0 
    # dist=20.0
    # angle_correction_factor = offset/dist * 360/( 2*np.pi) / 25.0
    # angle_correction_factor = .1146
    angle_correction_factor = .15
    
    for index in random_images_indexes:
        
        prob_position = np.random.choice(3, 1, p=[prob_center, prob_right, prob_left])

        if prob_position == 0:
            image = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            images_angles.append((image, angle))
            
        elif prob_position == 1:
            image = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - angle_correction_factor
            images_angles.append((image, angle))
            
        elif prob_position == 2:
            image = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + angle_correction_factor
            images_angles.append((image, angle))

    return images_angles
    
def generate_train_batch(batch_size = 64):
           
    while True:  
        image_batch = []
        angle_batch = []
        images_angles = generate_train_images(batch_size, prob_center = .4, prob_left = 0.3, prob_right = 0.3)
        
        for image_file, angle in images_angles:

            image = plt.imread('./data/data/' + image_file)
            
            #apply preprocessing
            image = process_image(image)
            
            #apply image augmentation
            augment_image_angle_data(image, angle, flip_prob = 0.5)

            #append image and angle to generated batch
            image_batch.append(image)
            angle_batch.append(angle)
            
        # assert len(image_batch) == batch_size, 'len(X_batch) == batch_size must be equal'
        yield np.array(image_batch), np.array(angle_batch)
        
def generate_validation_batch(batch_size = 64):
    
    while True:  
        image_batch = []
        angle_batch = []
        images_angles = generate_train_images(batch_size, prob_center = 0.4, prob_left = 0.3, prob_right = 0.3)
        
        for image_file, angle in images_angles:

            image = plt.imread('./data/data/' + image_file)
            
            #apply preprocessing, use only center images
            image = process_image(image)
            
            #do not apply image augmentation

            #append image and angle to generated batch
            image_batch.append(image)
            angle_batch.append(angle)
            
        # assert len(image_batch) == batch_size, 'len(X_batch) == batch_size must be equal'
        yield np.array(image_batch), np.array(angle_batch)
        
    
