# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:12:07 2021

@author: jiayi
"""


#loading dataset
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD


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

X_train = X_train
X_test = X_test
#prepare label data
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
    


####################################################################
def define_ini_model(X, y):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
	model.fit(X, y, epochs=50)
	return model
    
#########################################FAST######################
def FASTmodel():
# 	initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
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

def FACTselect(model,image, label,eps_fact):
    adv_x = FASTadv(model, image, label,eps_fact)
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


def FASTtrain(model,X_train,Y_train,eps_fast,eps_fact):
	ADV_X_fast = []
	ADV_X_fact = []
	ADV_Y_fact = []
	X_fact = []
	for i in range(len(X_train)):
		print(i)
		adv_x_fact,flag = FACTselect(model,np.array([X_train[i]]), Y_train[i],eps_fact)
		adv_x = FASTadv(model, np.array([X_train[i]]), Y_train[i],eps_fast)
		ADV_X_fast.append(np.array(adv_x[0],dtype = float))
		if not flag:
			ADV_X_fact.append(np.array(adv_x_fact[0],dtype = float))
			ADV_Y_fact.append(Y_train[i])
			X_fact.append(X_train[i])
	return ADV_X_fast,ADV_X_fact,ADV_Y_fact,X_fact

def FACTtrain(model,X_train,Y_train,eps_fast,eps_fact):
	ADV_X_fact = []
	ADV_Y_fact = []
	X_fact = []
	for i in range(len(X_train)):
		print(i)
		adv_x_fact,flag = FACTselect(model,np.array([X_train[i]]), Y_train[i],eps_fact)
		if not flag:
			ADV_X_fact.append(np.array(adv_x_fact[0],dtype = float))
			ADV_Y_fact.append(Y_train[i])
			X_fact.append(X_train[i])
	return ADV_X_fact,ADV_Y_fact,X_fact
    

#########################################PGD###############################
def PGDmodel():
# 	initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def PGDadv(model, image, label,eps,alpha):
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

def PGDtrain(model,X_train,Y_train,eps,alpha):	
	ADV_X_pgd = []
	for i in range(len(X_train)):
		print(i)
		adv_x= PGDadv(model,np.array([X_train[i]]), Y_train[i],eps,alpha)
		ADV_X_pgd.append(np.array(adv_x[0],dtype = float))
	return ADV_X_pgd


#########################################ATTACK#####################

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


#########################################MAIN######################
eps_fact = 2/255
eps_fast = 8/255
eps_pgd = 8/255
eps_attack = 4/255
alpha = 2/255
alpha_attack = 2/255


model_ini = define_ini_model(X_train, Y_train)

###########fast and fact########
model_fast = FASTmodel()
model_fact = FASTmodel()


# start_f_train = time.time()
# ADV_X_fast,ADV_X_fact,ADV_Y_fact,X_fact = FASTtrain(model_ini,X_train,Y_train,eps_fast,eps_fact)
# # ADV_X_fact,ADV_Y_fact,X_fact = FACTtrain(model_ini,X_train,Y_train,eps_fast,eps_fact)
# end_f_train = time.time()
# timeDiff_f_train = end_f_train - start_f_train
# ADV_Y_fast = []

# ADV_X_FAST = np.array(ADV_X_fast+list(np.array(X_train)))
# ADV_Y_FAST = np.array(list(Y_train)+list(Y_train))

# ADV_X_FACT = np.array(ADV_X_fact+X_fact)
# ADV_Y_FACT = np.array(ADV_Y_fact+ADV_Y_fact)

#########pgd###########
model_pgd = PGDmodel()

start_pgd_train = time.time()
ADV_X_pgd = PGDtrain(model_pgd,X_train,Y_train,eps_pgd,alpha)
end_pgd_train = time.time()
timeDiff_pgd_train = end_pgd_train - start_pgd_train
ADV_Y_pgd = []

ADV_X_PGD = np.array(ADV_X_pgd+list(np.array(X_train)))
ADV_Y_PGD = np.array(list(Y_train)+list(Y_train))



#48
####################################
#print the results
# start_fast = time.time()
# model_fast.fit(ADV_X_FAST, ADV_Y_FAST,epochs=100)
# end_fast = time.time()
# timeDiff_fast = end_fast - start_fast


# start_fact = time.time()
# model_fact.fit(ADV_X_FACT, ADV_Y_FACT,epochs=100)
# end_fact = time.time()
# timeDiff_fact = end_fact - start_fact


start_pgd = time.time()
model_pgd.fit(ADV_X_PGD, ADV_Y_PGD,epochs=100)
end_pgd = time.time()
timeDiff_pgd = end_pgd - start_pgd


# #time
# print("TIME###################")
# print("fast/fact data selection:",timeDiff_f_train)
# print("num of fast training:",len(ADV_X_FAST))
# print("fast training time:",timeDiff_fast)
# print("num of fact training:",len(ADV_X_FACT))
# print("fact training time:",timeDiff_fact)
# print("---------pgd---------")
print("pgd data selection:",timeDiff_pgd_train)
print("num of pgd training:",len(ADV_X_PGD))
print("pgd training time:",timeDiff_pgd)
   
#acc before attack
print("ACC before###################")
eva_test_ori = model_ini.evaluate(X_test, Y_test)
print("fast testing evaluation")
eva_test_fast = model_fast.evaluate(X_test, Y_test)
print("fact testing evaluation")
eva_test_fact = model_fact.evaluate(X_test, Y_test)
print("pgd testing evaluation")
eva_test_pgd = model_pgd.evaluate(X_test, Y_test)


########################APPLY ATTACK#############    
# =============================================================================
# def CHECK_Y(model,X_test,Y_test):
#     index_cor =[]
#     for i in range(len(X_test)):
#         y_pred = model(np.array([X_test[i]]))
#         y_pred_ind = tf.math.argmax(y_pred[0])
#         if y_pred_ind==tf.math.argmax(Y_test[i]):
#             index_cor.append(i)
#     return index_cor
#     
# def TEST_NEW(X_test,Y_test,index_cor):
#     X_test_new = []
#     Y_test_new = []
#     for i in index_cor:
#         X_test_new.append(X_test[i])
#         Y_test_new.append(Y_test[i])
#     return X_test_new,Y_test_new
# 
# def TEST_MODEL_pgd(model_ini,X_test,Y_test):
#     index_cor = CHECK_Y(model_ini,X_test,Y_test)
#     X_test_ini,Y_test_ini = TEST_NEW(X_test,Y_test,index_cor)
#     
#     pgd_X_test_ini = PGDattack(model_ini,X_test_ini,Y_test_ini,eps_attack,alpha_attack)
#     pgd_X_test_ini = np.array(pgd_X_test_ini)
#     pgd_Y_test_ini = np.array(Y_test_ini)
#     return pgd_X_test_ini,pgd_Y_test_ini
# 
# def TEST_MODEL_fgsm(model_ini,X_test,Y_test):
#     index_cor = CHECK_Y(model_ini,X_test,Y_test)
#     X_test_ini,Y_test_ini = TEST_NEW(X_test,Y_test,index_cor)
#     
#     pgd_X_test_ini = FGSMattack(model_ini,X_test_ini,Y_test_ini,eps_attack,alpha_attack)
#     pgd_X_test_ini = np.array(pgd_X_test_ini)
#     pgd_Y_test_ini = np.array(Y_test_ini)
#     return pgd_X_test_ini,pgd_Y_test_ini
# =============================================================================
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




pgd_X_test_ini,pgd_Y_test_ini = TEST_MODEL_pgd(model_ini,X_test,Y_test)


#acc after attack
print("ACC after###################")
eva_pgd_ini = model_ini.evaluate(pgd_X_test_ini,pgd_Y_test_ini)
print("fast testing evaluation")
eva_pgd_fast = model_fast.evaluate(pgd_X_test_ini,pgd_Y_test_ini)
print("fact testing evaluation")
eva_pgd_fact = model_fact.evaluate(pgd_X_test_ini,pgd_Y_test_ini)
print("pgd testing evaluation")
eva_pgd_pgd = model_pgd.evaluate(pgd_X_test_ini,pgd_Y_test_ini)
    


fgsm_X_test_ini,fgsm_Y_test_ini = TEST_MODEL_fgsm(model_ini,X_test,Y_test)


    
#acc after attack
print("ACC after###################")
eva_fgsm_ini = model_ini.evaluate(fgsm_X_test_ini,fgsm_Y_test_ini)
print("fast testing evaluation")
eva_fgsm_fast = model_fast.evaluate(fgsm_X_test_ini,fgsm_Y_test_ini)
print("fact testing evaluation")
eva_fgsm_fact = model_fact.evaluate(fgsm_X_test_ini,fgsm_Y_test_ini)
print("pgd testing evaluation")
eva_fgsm_pgd = model_pgd.evaluate(fgsm_X_test_ini,fgsm_Y_test_ini)
    
    
#################original###############
# =============================================================================
# model_ori = define_ini_model(X_train, Y_train)
# 
# #acc after attack
# print("ACC after###################")
# eva_test_ori = model_ori.evaluate(X_test, Y_test)
# print("ori pgd testing evaluation")
# eva_pgd_ori = model_ori.evaluate(pgd_X_test, pgd_Y_test)
# print("ori fgsm testing evaluation")
# eva_fgsm_ori = model_pgd.evaluate(fgsm_X_test, fgsm_Y_test)
# =============================================================================


# =============================================================================
# def define_model():
# 	model = Sequential()
# 	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
# 	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Dropout(0.2))
# 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Dropout(0.2))
# 	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# 	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Dropout(0.2))
# 	model.add(Flatten())
# 	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dropout(0.2))
# 	model.add(Dense(10, activation='softmax'))
# 	# compile model
# 	opt = SGD(lr=0.001, momentum=0.9)
# 	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 	return model
# 
# model_test = define_model()
# model_test.fit(X_train, Y_train, epochs=100)
# model_test.evaluate(X_test,Y_test)
# =============================================================================
