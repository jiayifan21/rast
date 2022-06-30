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


#loading library
cifar = datasets.cifar10 
(X_train, y_train), (X_test, y_test) = cifar.load_data()

#checking shape
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#define label

# label = ['Airplane', 'Automobile', 'Bird', 'Cat',
#                   'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# #plot image no 26 in training data with colorbar
# plt.figure()
# plt.imshow(X_train[26])
# plt.colorbar()

#rescaling it between 0 to 1
X_train = X_train/255.0
X_test = X_test/255.0
#prepare label data
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
    


####################################################################
def define_ini_model(X, y):
	model_ini = models.Sequential([
		layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
		layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
        
		layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
		layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
        
		layers.Flatten(),
		layers.Dense(256, activation='relu'),
		layers.Dense(256, activation='relu'),
		layers.Dense(10, activation='softmax')
		])
        
	model_ini.compile(optimizer='adam',
				   loss='categorical_crossentropy',
				   metrics=['accuracy'])
    
	model_ini.fit(X, y, epochs=50)
	return model_ini
    
#########################################FAST######################
def FASTmodel():
# 	initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
	model = models.Sequential([
		layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
		layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
        
		layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
		layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
        
		layers.Flatten(),
		layers.Dense(256, activation='relu'),
		layers.Dense(256, activation='relu'),
		layers.Dense(10, activation='softmax')
		])
        
	model.compile(optimizer='adam',
				   loss='categorical_crossentropy',
				   metrics=['accuracy'])
	return model

def FASTadv(model, image, label,eps):
    input_image = tf.cast(image, tf.float32)
    input_label = tf.cast(label, tf.int64)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction[0])
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    #add gradient
    delta = tf.random.uniform(shape=tf.shape(input_image), minval=-eps, maxval=eps)
    delta = delta + eps*signed_grad
    delta = tf.clip_by_value(delta, -eps, eps)
    adv_x = input_image + delta
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x

def FASTselect(model,image, label,eps):
    adv_x = FASTadv(model, image, label,eps)
    adv_pred_y = model(adv_x)
    adv_pred_y = tf.math.argmax(adv_pred_y[0])
    # print("@@@@@@@@@@@@@@@")
    # print(adv_pred_y)
    pred_y = model(image)
    pred_y = tf.math.argmax(pred_y[0])
    # print(pred_y)
    if np.array(adv_pred_y) == np.array(pred_y):
        flag = True
    else: 
        flag = False
    return adv_x,flag

def FASTtrain(model,X_train,Y_train,eps):
	ADV_X_fast = []
	ADV_X_fact = []
	ADV_Y_fact = []
	for i in range(len(X_train)):
		print(i)
		adv_x,flag = FASTselect(model,np.array([X_train[i]]), Y_train[i],eps)
		ADV_X_fast.append(np.array(adv_x[0],dtype = float))
		if not flag:
			ADV_X_fact.append(np.array(adv_x[0],dtype = float))
			ADV_Y_fact.append(Y_train[i])
	return ADV_X_fast,ADV_X_fact,ADV_Y_fact

#########################################PGD###############################
def PGDmodel():
# 	initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
 	model = models.Sequential([
		layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
		layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
        
		layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
		layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
        
		layers.Flatten(),
		layers.Dense(256, activation='relu'),
		layers.Dense(256, activation='relu'),
		layers.Dense(10, activation='softmax')
		])
        
 	model.compile(optimizer='adam',
 				   loss='categorical_crossentropy',
 				   metrics=['accuracy'])
 	return model

def PGDadv(model, image, label,eps,alpha):
    input_image = tf.cast(image, tf.float32)
    input_label = tf.cast(label, tf.int64)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction[0])
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    #add gradient
    delta = tf.zeros(tf.shape(input_image), tf.int32)
    for i in range(8):
        delta = delta + alpha*signed_grad
        delta = tf.clip_by_value(delta, -eps, eps)
    adv_x = input_image + delta
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x

def PGDselect(model,image, label,eps):
    adv_x = PGDadv(model, image, label,eps)
    adv_pred_y = model(adv_x)
    adv_pred_y = tf.math.argmax(adv_pred_y[0])
    # print("@@@@@@@@@@@@@@@")
    # print(adv_pred_y)
    pred_y = model(image)
    pred_y = tf.math.argmax(pred_y[0])
    # print(pred_y)
    if np.array(adv_pred_y) == np.array(pred_y):
        flag = True
    else: 
        flag = False
    return adv_x,flag

def PGDtrain(model,X_train,Y_train,eps):	
	ADV_X_pgd = []
	for i in range(len(X_train)):
		print(i)
		adv_x= PGDadv(model,np.array([X_train[i]]), Y_train[i],eps)
		ADV_X_pgd.append(np.array(adv_x[0],dtype = float))
	return ADV_X_pgd


#########################################MAIN######################
eps = 12/255
eps_pgd = 8/255
alpha = 2/255


model_ini = define_ini_model(X_train[:10000], Y_train[:10000])

###########fast and fact
model_fast = FASTmodel()
model_fact = FASTmodel()

ADV_X_fast,ADV_X_fact,ADV_Y_fact = FASTtrain(model_ini,X_train,Y_train,eps)
ADV_Y_fast = []

ADV_X_FAST = np.array(ADV_X_fast+list(np.array(X_train)))
ADV_Y_FAST = np.array(list(Y_train)+list(Y_train))

ADV_X_FACT = np.array(ADV_X_fact+list(X_train))
ADV_Y_FACT = np.array(ADV_Y_fact+list(Y_train))

#########pgd
model_pgd = PGDmodel()
ADV_X_pgd = PGDtrain(model_ini,X_train,Y_train,eps)
ADV_Y_pgd = []

ADV_X_PGD = np.array(ADV_X_pgd+list(np.array(X_train)))
ADV_Y_PGD = np.array(list(Y_train)+list(Y_train))


####################################
#print the results
start_fast = time.time()
model_fast.fit(ADV_X_FAST, ADV_Y_FAST,epochs=60)
end_fast = time.time()
timeDiff_fast = end_fast - start_fast


start_fact = time.time()
model_fact.fit(ADV_X_FACT, ADV_Y_FACT,epochs=60)
end_fact = time.time()
timeDiff_fact = end_fact - start_fact


start_pgd = time.time()
model_pgd.fit(ADV_X_PGD, ADV_Y_PGD,epochs=60)
end_pgd = time.time()
timeDiff_pgd = end_pgd - start_pgd


#time
print("TIME###################")
print("num of fast training:",len(ADV_X_FAST))
print("fast training time:",timeDiff_fast)
print("num of fact training:",len(ADV_X_FACT))
print("fact training time:",timeDiff_fact)
print("num of pgd training:",len(ADV_X_PGD))
print("pgd training time:",timeDiff_pgd)
   
#acc before attack
print("ACC before###################")
print("fast testing evaluation")
eva_test_fast = model_fast.evaluate(X_test, Y_test)
print("fact testing evaluation")
eva_test_fact = model_fact.evaluate(X_test, Y_test)
print("pgd testing evaluation")
eva_test_pgd = model_pgd.evaluate(X_test, Y_test)

  

    
    
    
    
    
    
    
    
    
    
    