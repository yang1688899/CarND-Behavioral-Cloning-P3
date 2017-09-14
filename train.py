# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:35:25 2017

@author: yang
"""

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

#define a generator
def generator(samples, batch_size=32):
    samples_size = len(samples)
    while 1:
        shuffle(samples)
        for offset in (0,batch_size,samples_size):
            batch_samples = samples[offset:offset+batch_size]
            
            imgs = []
            measurements = []
            
        

        for sample in batch_samples:
            current_center_paths = [get_current_path(path) for path in sample[:3]]
            img_samples = [cv2.imread(path) for path in current_center_paths]
            
            measurement = float(line[3])
            correction = 0.2
            measurement_left = measurement + correction
            measurement_right = measurement - correction
            measurement_samples = [measurement,measurement_left,measurement_right]
            
            imgs.extend(img_samples)
            measurements.extend(measurement_samples)

    
        for i in range(len(imgs)):
            image_flipped = np.fliplr(imgs[i])
            measurement_flipped = -measurements[i]
            imgs.append(image_flipped)
            measurements.append(measurement_flipped)
    
    
        X_train = np.array(imgs)
        y_train = np.array(measurements)
        print(X_train.shape[1:])
        yield X_train,y_train
 
def get_current_path(path):
    fileName = path.split('\\')[-1]
    currentPath = '../my_data/IMG/' + fileName
    return currentPath
    
lines = []
with open('../my_data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    next(reader)
    for line in reader:
        lines.append(line)
#split the train and validate set        
train_samples, validate_samples = train_test_split(lines,test_size=0.2)

train_generator = generator(train_samples,batch_size=256)
validate_generator = generator(validate_samples,batch_size=256)
        

model = Sequential()

model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=[160, 320, 3]))
model.add(Lambda(lambda x: x/255. - 0.5))
model.add(Conv2D(filters=6,kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=16,kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))

model.add(Dense(86))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator,samples_per_epoch=len(train_samples)*6,validation_data=\
          validate_generator,nb_val_samples=len(validate_samples)*6,nb_epoch=5)
model.save('model.h5')
exit()
    