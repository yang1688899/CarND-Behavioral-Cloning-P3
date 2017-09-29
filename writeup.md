#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image/bias_iamge.png
[image2]: ./image/image2.png "Grayscaling"
[image3]: ./image/image3.png "Recovery Image"
[image4]: ./image/image4.jpg "Recovery Image"
[image5]: ./image/image5.jpg "Recovery Image"
[image6]: ./image/image6.jpg "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model consisted 3 Convolution layers with depth of 6,16,32 and a 2x2 Max pooling layer after each convolution layers, and 3 fully connected layer:
```
model = Sequential()

model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=[160, 320, 3]))
model.add(Lambda(lambda x: x/255. - 0.5))
model.add(Conv2D(filters=6,kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=16,kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Dropout(0.5))

model.add(Conv2D(filters=32,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(86))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
```

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

First use a convolution neural network model similar to the LeNet architecture, just wanted to see how this architecture going to perform on such problem. After I split my image and steering angle data into a training and validation set. I train my model with the training data and found that above 5 epoch the squared error on the validation set start to increase instead of decrease. This indicated that I must overfitting my data. So I add 3 dropout layer between the fully connected layer.After I preprocess the data, the model have been trained whitout overfitting. But this model still not capabale to drive the car through the entire track whitout break out off the road. So I used the more powerful network-the navidia network, by adding a maxpoolinng layer for each convolution layer and a dropout layer after first 3 convolution layer and use relu as activation function for each layer.After i train my model used the preprocessed data, I got a model that can drive the vehicle is able to drive autonomously around the track without leaving the road!

####2. Final Model Architecture

The final model consisted 5 Convolution layers with depth of 24,36,48,64,64 and a 2x2 Max pooling layer after each convolution layers, and 3 fully connected layer.In order to avoids overfitting, also add a dropout layer between this layer execpt the first 3 convolution layers:
```
model = Sequential()

model.add(Conv2D(filters=24,kernel_size=(5,5),input_shape=[399, 600, 3]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=36,kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=48,kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Dropout(0.5))

model.add(Conv2D(filters=64,kernel_size=(3,3)))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Conv2D(filters=64,kernel_size=(3,3)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(80))
model.add(Activation('softmax'))

```


####3. Creation of the Training Set & Training Process
First I used the provided data to train the network, which provided a terrible result. So i decide look deeper into the data. And find that the data is very dias. So I have to collect more data on my own.

At each frame, 3 images is taken by 3 cameras in difference places:
![alt text][image4]
the center image

![alt text][image5]
the left image

![alt text][image6]
the right image

I used all three images taken for each frame in my training, and add(subtract)0.2 steering angle for the left and right image.

In order to capture good driving behavior, I first recorded two laps on track one using center lane driving, then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn. And add all my collected data with the data Udacity provided, I got a data set like this:
![alt text][image1]
Seems the data set have too little hard turn data.
So I do more recovering driving to capture more hard turn data. After a prolong data collecting process, I got a data set look like this:
![alt text][image2]
Seems that the way I drive the car the collect more data is not quit right, I do get more hard turning data, but even more go straigt data......
So i decide to add code a filter to randomly filter out a lot the data that steering angle is too small:
```
for line in reader:
        if float(line[3])>0.0 and float(line[3])<0.1:
            if round(random.random()*0.58):
                lines.append(line)
        elif float(line[3])>-0.2 and float(line[3])<0.2:
            if round(random.random()):
                lines.append(line)
        else:
            lines.append(line)
```
Finilly I got a data set look like this:
![alt text][image3]

Still a bias data, but mush better, good enough to train a model that could pass track 1!




