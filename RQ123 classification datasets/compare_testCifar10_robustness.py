# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 04:39:34 2021

@author: jiayi
"""

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
import random


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

eps_attack = 4/255
alpha_attack = 2/255

###################################DEFINE FUNCTIONS###############################
# define cnn model
def define_model(X, y):
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


def define_ini_model(X, y):
    model_ini = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
        
    model_ini.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    model_ini.fit(X, y, epochs=50)
    return model_ini


def BSS_select(model, image, T):
	#input_image = tf.cast(image, tf.float32)
	prediction = model(image)
	Y = prediction.numpy()
	Y.sort()
	Y_max = Y[0][-1]
	Y_secmax = Y[0][-2]
	ratio = Y_max/Y_secmax
	if ratio <= T:
		flag = False
	else:
		flag = True
	return flag


def ADV_samp(model, image, label,eps):
    input_image = tf.cast(image, tf.float32)
    input_label = tf.cast(label, tf.int64)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        # predict_y = tf.math.argmax(prediction[0])
        # predict_y = tf.expand_dims(predict_y,axis=0)
        #predict_y = tf.cast(predict_y, tf.float32)
        # print("!!!!!!!!!!!!!!!!!!")
        # print(input_label)
        # print(prediction[0])
        loss = loss_object(input_label, prediction[0])
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    adv_x = input_image + eps*signed_grad
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x
    


def ADV_select(model,image, label,eps):
    adv_x = ADV_samp(model, image, label,eps)
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
    return flag

########attack########################################
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

def PGDattack_adv(model, image, label,eps,alpha):
    input_image = tf.cast(image, tf.float32)
    input_label = tf.cast(label, tf.int64)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    #add gradient
    delta = tf.random.uniform(shape=tf.shape(input_image), minval=-eps, maxval=eps)
    adv_x = input_image
    for i in range(8):
        with tf.GradientTape() as tape:
            tape.watch(adv_x)
            prediction = model(adv_x)
            loss = loss_object(input_label, prediction[0])
            
        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, adv_x)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        
        delta = delta + alpha*signed_grad
        delta = tf.clip_by_value(delta, -eps, eps)
    adv_x = adv_x + delta
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x


def PGDattack(model,X_test,Y_test,eps,alpha):	
	ADV_X_pgd = []
	for i in range(len(X_test)):
		print(i)
		adv_x= PGDattack_adv(model,np.array([X_test[i]]), Y_test[i],eps,alpha)
		ADV_X_pgd.append(np.array(adv_x[0],dtype = float))
	return ADV_X_pgd

def FGSMattack_adv(model, image, label,eps,alpha):
    input_image = tf.cast(image, tf.float32)
    input_label = tf.cast(label, tf.int64)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction[0])
        # loss = loss_object(input_label, input_image)
        
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


def FGSMattack(model,X_test,Y_test,eps,alpha):	
	ADV_X_fgsm = []
	for i in range(len(X_test)):
		print(i)
		adv_x= FGSMattack_adv(model,np.array([X_test[i]]), Y_test[i],eps,alpha)
		ADV_X_fgsm.append(np.array(adv_x[0],dtype = float))
	return ADV_X_fgsm

def CHECK_Y(model,X_test,Y_test):
    index_cor =[]
    for i in range(len(X_test)):
        y_pred = model(np.array([X_test[i]]))
        y_pred_ind = tf.math.argmax(y_pred[0])
        if y_pred_ind != tf.math.argmax(Y_test[i]):
            index_cor.append(i)
    return index_cor
    
def TEST_NEW(X_test,Y_test,index_cor):
    X_test_new = []
    Y_test_new = []
    for i in index_cor:
        X_test_new.append(X_test[i])
        Y_test_new.append(Y_test[i])
    return X_test_new,Y_test_new

def TEST_MODEL_pgd(model_ini,X_test,Y_test):
    X_test_ini = PGDattack(model_ini,X_test,Y_test,eps_attack,alpha_attack)
    index_cor = CHECK_Y(model_ini,X_test_ini,Y_test)
    pgd_X_test_ini,Y_test_ini = TEST_NEW(X_test_ini,Y_test,index_cor)
    
    pgd_X_test_ini = np.array(pgd_X_test_ini)
    pgd_Y_test_ini = np.array(Y_test_ini)
    return pgd_X_test_ini,pgd_Y_test_ini

def TEST_MODEL_fgsm(model_ini,X_test,Y_test):
    X_test_ini = FGSMattack(model_ini,X_test,Y_test,eps_attack,alpha_attack)
    index_cor = CHECK_Y(model_ini,X_test_ini,Y_test)
    fgsm_X_test_ini,Y_test_ini = TEST_NEW(X_test_ini,Y_test,index_cor)
    fgsm_X_test_ini = np.array(fgsm_X_test_ini)
    fgsm_Y_test_ini = np.array(Y_test_ini)
    return fgsm_X_test_ini,fgsm_Y_test_ini


#########################################MIAN LOGIC#####################3

#train initial model for data selection
model_ini = define_model(X_train[:10000], Y_train[:10000])

#bench mark
start_ini = time.time()
model_all = define_model(X_train, Y_train)
end_ini = time.time()
timeDiff_ini = end_ini - start_ini

#generate attack data
pgd_X_test_ini,pgd_Y_test_ini = TEST_MODEL_pgd(model_ini,X_test,Y_test)
fgsm_X_test_ini,fgsm_Y_test_ini = TEST_MODEL_fgsm(model_ini,X_test,Y_test)

#benchmark
eva_pgd_ini = model_ini.evaluate(pgd_X_test_ini,pgd_Y_test_ini)[1]
eva_pgd_all = model_all.evaluate(pgd_X_test_ini,pgd_Y_test_ini)[1]
eva_fgsm_ini = model_ini.evaluate(fgsm_X_test_ini,fgsm_Y_test_ini)[1]
eva_fgsm_all = model_all.evaluate(fgsm_X_test_ini,fgsm_Y_test_ini)[1]

#define parameters
model = model_ini
# epss = [0.001,0.002,0.003,0.004,0.005,0.01,0.05,0.1,0.2]
# epss = [0.001,0.002,0.004,0.02,0.3]
epss = [0.001,0.002,0.004,0.02,0.3]
T=[1.5,5.0,10.0,1000,10000]
R=[2000,5000,10000,20000,30000]


adv_train = []
adv_test = []
adv_time = []
adv_num = []
adv_pgd = []
adv_fgsm = []

bss_train = []
bss_test = []
bss_time = []
bss_num = []
bss_pgd = []
bss_fgsm = []

##########################select data############
#for n in range(5):
for n in range(len(epss)):
# for t in T: #!
    image_keep_adv = []
    label_keep_adv = []
    index_keep_adv = []
    image_keep_bss = []
    label_keep_bss = []
    index_keep_bss = []
    #select data
    for i in range(len(X_train)): 
        print(i)
        image = np.array([X_train[i]])
        label = Y_train[i]   
        
        FLAG_adv = ADV_select(model, image, label, epss[n])
        FLAG_bss = BSS_select(model, image, T[n]) #!
        
        if not FLAG_adv:
            image_keep_adv.append(X_train[i])
            label_keep_adv.append(Y_train[i])
            index_keep_adv.append(i)

        if not FLAG_bss:
            image_keep_bss.append(X_train[i])
            label_keep_bss.append(Y_train[i])
            index_keep_bss.append(i)

    # # # organize delected data; slt -> select
    slt_X_train_adv = np.array(image_keep_adv[:R[n]])
    slt_y_train_adv = np.array(label_keep_adv[:R[n]])#[np.argmax(item) for item in label_keep])

    slt_X_train_bss = np.array(image_keep_bss[:R[n]])
    slt_y_train_bss = np.array(label_keep_bss[:R[n]])#[np.argmax(item) for item in label_keep])

    
    #calculate time and compare
    model_adv = define_model(slt_X_train_adv, slt_y_train_adv)
    model_bss = define_model(slt_X_train_bss, slt_y_train_bss)

    #########apply attacks
    print("ACC after pgd###################")
    eva_pgd_adv = model_adv.evaluate(pgd_X_test_ini,pgd_Y_test_ini)[1]
    eva_pgd_bss = model_bss.evaluate(pgd_X_test_ini,pgd_Y_test_ini)[1]

    print("ACC after fgsm###################")
    eva_fgsm_adv = model_adv.evaluate(fgsm_X_test_ini,fgsm_Y_test_ini)[1]
    eva_fgsm_bss = model_bss.evaluate(fgsm_X_test_ini,fgsm_Y_test_ini)[1]

    print("After testing evaluation")
    eva_test_adv = model_adv.evaluate(X_test, Y_test)
    print("After testing evaluation")
    eva_test_bss = model_bss.evaluate(X_test, Y_test)    

    
    # list_train.append(eva_train_slt)
    adv_test.append(eva_test_adv)
    adv_num.append(len(slt_y_train_adv))
    adv_pgd.append(eva_pgd_adv)
    adv_fgsm.append(eva_fgsm_adv)

    bss_test.append(eva_test_bss)
    bss_num.append(len(slt_y_train_bss))
    bss_pgd.append(eva_pgd_bss)
    bss_fgsm.append(eva_fgsm_bss)

###########train models################






print("Initial training evaluation")
eva_train_ini = model_all.evaluate(X_train, Y_train)
print("Initial testing evaluation")
eva_test_all = model_all.evaluate(X_test, Y_test)
print("Initial training time:",timeDiff_ini)