## **车辆行为复制（Behavioral Cloning）** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**实现步骤**

* 使用模拟器收集训练
* 使用Keras搭建CNN，训练校验模型
* 在模拟器上测试模型表现

[//]: # (Image References)

[image1]: ./image/bias_iamge.png
[image2]: ./image/image2.png "Grayscaling"
[image3]: ./image/image3.png "Recovery Image"
[image4]: ./image/image4.jpg "Recovery Image"
[image5]: ./image/image5.jpg "Recovery Image"
[image6]: ./image/image6.jpg "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


#### 使用模拟器收集训练数据

使用udacity提供的模拟器收集训练数据，训练数据特征(features)为车前部摄像头捕捉到的原始像素图，标签(lable)为汽车的方向操控命令。模拟器汽车使用左中右三个摄像头捕捉每一帧图片，以下为同一帧下不同摄像头捕捉的图片:

中：

![alt text][image4]

左：

![alt text][image5]

右：

![alt text][image6]

#### 使用Keras搭建CNN，训练校验模型
最终的模型结构如下:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 image   							    | 
| Convolution 5x5     	| 1x1 stride, valid padding, 24 filters 	|
| Relu					|												|
| Max pooling 2x2	      	| 2x2 stride				    |
| Convolution 5x5	    | 1x1 stride, valid padding, 36 filters   |
| Relu          		|       									    |
|Max pooling     		| 2x2 stride					|
| Convolution 5x5	    | 1x1 stride, valid padding, 48 filters   |
| Relu          		|       									    |
|Max pooling     		| 2x2 stride					|
|Dropout     		| 0.5					|
| Convolution 5x5	    | 1x1 stride, valid padding, 64 filters   |
| Relu          		|       									    |
|Dropout     		| 0.5					|
| Convolution 5x5	    | 1x1 stride, valid padding, 64 filters   |
| Relu          		|       									    |
|Dropout     		| 0.5					|
|Flatten				| 									|
|Fully Connected    	| outputs 100					                |
| Relu          		|       									    |
|Dropout     		| 0.5					|
|Fully Connected    	| outputs 50					                |
| Relu          		|       									    |
|Dropout     		| 0.5					|
|Fully Connected    	| outputs 10					                |
| Relu          		|       									    |
|Dropout     		| 0.5					|
|Fully Connected    	| outputs 1					                |

以下为实现代码:

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
训练使用的优化器(optimizer)为adam,损失函数(loss function)为mse,一共进行5 epoch训练
```
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator,steps_per_epoch=len(train_samples)/batch_size,validation_data=\
          validate_generator,nb_val_samples=len(validate_samples)/batch_size,nb_epoch=5)
```
#### 在模拟器上测试模型表现

以下为在模拟器跑道1表现：
[跑道1结果视频](./run1.mp4)

以下为模拟器在跑道2表现：
[跑道2结果视频](./run2.mp4)

