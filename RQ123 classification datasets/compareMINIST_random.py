# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:12:07 2021

@author: jiayi
"""


#loading dataset
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import random



#loading library
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#checking shape
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#define label

# #plot image no 26 in training data with colorbar
# plt.figure()
# plt.imshow(X_train[26])
# plt.colorbar()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#rescaling it between 0 to 1
X_train = x_train#/255.0
X_test = x_test#/255.0
#prepare label data
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)


###################################DEFINE FUNCTIONS###############################
# define cnn model
def define_model(X, y):
	model = models.Sequential([
		layers.Conv2D(32, kernel_size=(3,3), activation = "relu", input_shape=(28, 28, 1)),
		layers.Conv2D(32, kernel_size=(3,3), activation = "relu"),
		layers.MaxPooling2D(pool_size=(2, 2)),
		layers.Conv2D(64, kernel_size=(3,3), activation = "relu"),
		layers.Conv2D(64, kernel_size=(3,3), activation = "relu"),
		layers.MaxPooling2D(pool_size=(2, 2)),
		layers.Flatten(), # Flattening the 2D arrays for fully connected layers
		
		layers.Dense(200, activation="relu"),
		layers.Dense(10,activation="softmax")])
        
	model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
	
	model.fit(X, y, epochs=20)
	return model



# model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None,input_tensor=None,input_shape=(32,32,3), pooling=None, classes=10)


##################################BSS#########################
# def BSS_select(model, image, T):
# 	#input_image = tf.cast(image, tf.float32)
# 	prediction = model(image)
# 	Y = prediction.numpy()
# 	Y.sort()
# 	Y_max = Y[0][-1]
# 	Y_secmax = Y[0][-2]
# 	ratio = Y_max/Y_secmax
# 	if ratio <= T:
# 		flag = False
# 	else:
# 		flag = True
# 	return flag
    
    
    
#########################################MIAN LOGIC#####################3

#train initial model for data selection
model_ini = define_model(X_train[:15000], Y_train[:15000])




#define parameters
model = model_ini
#epss = [0.001,0.005,0.01,0.05,0.1,0.2]
# T=[1.5,2.0,5.0,10.0,100,1000,10000]
R=[500,1500,2500,20000]


list_train = []
list_test = []
list_time = []
list_num = []


#for eps in epss:
# for t in T:
for r in R:
    image_keep = []
    label_keep = []
    index_keep = []
    #select data
    # for i in range(len(X_train)): 
    #     print(i)
    #     image = np.array([X_train[i]])
    #     label = Y_train[i]   
        
        # FLAG = BSS_select(model, image, t)
        
        # if not FLAG:
        #     image_keep.append(X_train[i])
        #     label_keep.append(Y_train[i])
        #     index_keep.append(i)
        
        
    # random
    slt_index = random.sample(list(enumerate(X_train)),r)
    slt_INDEX = []
    slt_x_train = []
    slt_y_train = []
    for ind in slt_index:
        slt_INDEX.append(ind[0])
        slt_x_train.append(ind[1])
        slt_y_train.append(Y_train[ind[0]])
        
    slt_X_train = np.array(slt_x_train)
    slt_y_train = np.array(slt_y_train)#[np.argmax(item) for item in label_keep])
    
    
    
    #select the first r data points
    #image_keep = [X_train[:r]]
    #label_keep = [Y_train[:r]]
    # slt_X_train = np.array(image_keep)
    # slt_y_train = np.array(label_keep)#[np.argmax(item) for item in label_keep])
    
    #calculate time and compare
    start_slt = time.time()
    model_slt = define_model(slt_X_train, slt_y_train)
    end_slt = time.time()
    timeDiff_slt = end_slt - start_slt
    
    #print the resutls
    print("After training evaluation")
    eva_train_slt = model_slt.evaluate(slt_X_train, slt_y_train)
    print("After testing evaluation")
    eva_test_slt = model_slt.evaluate(X_test, Y_test)
    
    
    print("Optimized training time:",timeDiff_slt)
    
    list_train.append(eva_train_slt)
    list_test.append(eva_test_slt)
    list_time.append(timeDiff_slt)
    list_num.append(len(slt_y_train))



#print the resutls
start_ini = time.time()
model_all = define_model(X_train, Y_train)
end_ini = time.time()
timeDiff_ini = end_ini - start_ini

print("Initial training evaluation")
eva_train_ini = model_all.evaluate(X_train, Y_train)
print("Initial testing evaluation")
eva_test_ini = model_all.evaluate(X_test, Y_test)
print("Initial training time:",timeDiff_ini)







########################PLOT!!!!!!!!!!!!!!!!!!!!
# T=[0,1.2,1.5,2.0]
R=[60000,500,1500,2500,20000]
li_train = [eva_train_ini] + list_train
li_test = [eva_test_ini] + list_test
li_time = [timeDiff_ini] + list_time
li_num = [60000] + list_num


# plt.bar(np.array(R).astype('str'), li_num)
# plt.title('Number of selected data')
# plt.grid(True)
# for i in range(len(li_num)):
#     plt.annotate(str(li_num[i]), xy=(i,li_num[i]), ha='center', va='bottom')
# plt.show()

plt.plot(np.array(R).astype('str'), li_time,'--bo', label='line with marker')
plt.title('Training time (s)')
plt.grid(True)
for i in range(len(li_time)):
    plt.annotate(str(int(li_time[i])), xy=(i,li_time[i]), ha='left', va='top')
plt.show()


# plt.plot(np.array(R).astype('str'), np.array(li_train),'--bo', label='line with marker')
# plt.title('Training Accuracy')
# plt.grid(True)
# plt.ylim([0,1])
# for i in range(len(li_time)):
#     plt.annotate(str(round(np.array(li_train)[i],2)), xy=(i,np.array(li_train)[i]), ha='right', va='top')
# plt.show()

plt.plot(np.array(R).astype('str'), np.array(li_test)[:,1],'--bo', label='line with marker')
plt.title('Testing Accuracy')
plt.grid(True)
plt.ylim([0,1])
for i in range(len(li_time)):
    plt.annotate(str(round(np.array(li_test)[:,1][i],2)), xy=(i,np.array(li_test)[:,1][i]), ha='left', va='top')
plt.show()

    
    
    
    
    
    
    
    
    