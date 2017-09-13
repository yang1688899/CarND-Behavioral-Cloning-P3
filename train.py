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
 
def get_current_path(path):
    fileName = path.split('/')[-1]
    currentPath = '../my_data/IMG/' + fileName
    return currentPath
    
lines = []
with open('../my_data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    next(reader)
    for line in reader:
        lines.append(line)
        
    
imgs = []
measurements = []
for line in lines:

    current_center_paths = [get_current_path(path) for path in line[:3]]
    img_samples = [cv2.imread(path) for path in current_center_paths]
    
    measurement = float(line[3])
    correction = 0.2
    measurement_left = measurement + correction
    measurement_right = measurement - correction
    measurement_samples = [measurement,measurement_left,measurement_right]
    
    imgs.extend(img_samples)
    measurements.extend(measurement_samples)
print(len(imgs))
print(len(measurements))
    
for i in range(len(imgs)):
    image_flipped = np.fliplr(imgs[i])
    measurement_flipped = -measurements[i]
    imgs.append(image_flipped)
    measurements.append(measurement_flipped)
    
print(len(imgs))
print(len(measurements))
    
X_train = np.array(imgs)
y_train = np.array(measurements)

input_shape = X_train.shape[1:]
model = Sequential()

model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=input_shape))
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
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=5,batch_size=128)
model.save('model.h5')
exit()
    