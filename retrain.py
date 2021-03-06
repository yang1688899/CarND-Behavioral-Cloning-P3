
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:04:29 2017

@author: yang
"""

from keras.models import load_model
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Activation,Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

#define a generator
def generator(samples, batch_size=32):
    samples_size = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0,samples_size,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            imgs = []
            measurements = []

            for sample in batch_samples:
                current_center_paths = [get_current_path(path) for path in sample[:3]]
                img_samples = [cv2.imread(path) for path in current_center_paths]
                
                measurement = float(sample[3])
                correction = 0.2
                measurement_left = measurement + correction
                measurement_right = measurement - correction
                measurement_samples = [measurement,measurement_left,measurement_right]
                
                imgs.extend(img_samples)
                measurements.extend(measurement_samples)
        
            for i in range(len(imgs)):
                if round(random.random()):
                    image_flipped = np.fliplr(imgs[i])
                    measurement_flipped = -measurements[i]
                    imgs.append(image_flipped)
                    measurements.append(measurement_flipped)
    
    
            X_train = np.array(imgs)
            y_train = np.array(measurements)
            yield X_train,y_train
 
def get_current_path(path):
    path = path.replace('/','\\')
    fileName = path.split('\\')[-1]
    currentPath = '../data/IMG/' + fileName
    return currentPath
    
lines = []
with open('../data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    next(reader)
    for line in reader:
        if float(line[3])>0.0 and float(line[3])<0.1:
            if round(random.random()*0.58):
                lines.append(line)
        elif float(line[3])>-0.2 and float(line[3])<0.2:
            if round(random.random()):
                lines.append(line)
        else:
            lines.append(line)
#split the train and validate set        


batch_size = 32
train_generator = generator(train_samples,batch_size=batch_size)
validate_generator = generator(validate_samples,batch_size=batch_size) 

model = load_model('model.h5')

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator,steps_per_epoch=len(train_samples)/batch_size,validation_data=\
          validate_generator,nb_val_samples=len(validate_samples)/batch_size,nb_epoch=1)
model.save('model.h5')
