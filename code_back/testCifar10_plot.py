# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:12:07 2021

@author: jiayi
"""


#loading dataset
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time



#loading library
cifar = datasets.cifar10 
(X_train, y_train), (X_test, y_test) = cifar.load_data()

#checking shape
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#define label

label = ['Airplane', 'Automobile', 'Bird', 'Cat',
                  'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

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


###################################DEFINE FUNCTIONS###############################
# define cnn model
def define_model(X_train, Y_train):
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
    
    model_ini.fit(X_train, Y_train, epochs=50)
    return model_ini



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
    pred_y = model(image)
    pred_y = tf.math.argmax(pred_y[0])
    if np.array(adv_pred_y) == np.array(pred_y):
        flag = True
    else: 
        flag = False
        # print("!!!!!!!!!!!")
        # print(adv_pred_y)
        # print(pred_y)
    return flag
        

def RETRAIN(eps,X_train,Y_train):
    #define parameters
    model = model_ini
    image_keep = []
    label_keep = []
    index_keep = []
    
    
    #select data
    for i in range(len(Y_train)): 
        print(i)
        image = np.array([X_train[i]])
        label = Y_train[i]   
        
        FLAG = ADV_select(model, image, label, eps)
        
        if not FLAG:
            print(FLAG)
            image_keep.append(X_train[i])
            label_keep.append(Y_train[i])
            index_keep.append(i)
        
     
    #organize delected data; slt -> select
    # slt_X_train, slt_X_test, slt_Y_train, slt_Y_test = train_test_split(np.array(image_keep), np.array(label_keep), test_size=0.2, random_state=42)
    # slt_y_train = np.array([np.argmax(item) for item in slt_Y_train])
    # slt_y_test = np.array([np.argmax(item) for item in slt_Y_test])
    slt_X_train = np.array(image_keep)
    slt_y_train = np.array(label_keep)#[np.argmax(item) for item in label_keep])

    
    #calculate time and compare
    if len(slt_X_train) !=0:
        start_slt = time.time()
        model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(slt_X_train, slt_y_train, epochs=50)
        end_slt = time.time()
    else:
        start_slt = time.time()
        end_slt = start_slt
    
    print("tesssssssssssssssssst")
    timeDiff_slt = end_slt - start_slt
    num_data = len(label_keep)
    eva_train = model.evaluate(X_train, Y_train)
    eva_test = model.evaluate(X_test, Y_test)
    
    return num_data, timeDiff_slt, eva_train, eva_test

    
    
#########################################MIAN LOGIC#####################3


#train initial model for data selection
X_new, X_old, Y_new, Y_old = train_test_split(np.array(X_train), np.array(Y_train), test_size=0.2, random_state=42)
# y_old = np.array([np.argmax(item) for item in Y_old])
# x_old = np.array([np.argmax(item) for item in X_old])

model_ini = define_model(X_old, Y_old)
    


# =============================================================================
# #update the model with all data
# start_ini = time.time()
# 
# model_all = model_ini
# model_all.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model_all.fit(X_train, y_train, epochs=20)
# 
# end_ini = time.time()
# timeDiff_ini = end_ini - start_ini
# =============================================================================





list_eps = [0.001,0.005,0.01,0.05,0.1,0.2]
list_per = []
list_num = []
list_time = []
list_train_acc = []
list_train_loss = []
list_test_acc = []
list_test_loss = []

for eps in list_eps:
    num_data, timeDiff_slt, eva_train, eva_test = RETRAIN(eps,X_train,Y_train)
    list_num.append(num_data)
    list_per.append(round(num_data/len(Y_train),2))
    list_time.append(timeDiff_slt)
    list_train_acc.append(eva_train[1])
    list_train_loss.append(eva_train[0])
    list_test_acc.append(eva_test[1])
    list_test_loss.append(eva_test[0])    

    print(eva_test)



plt.plot(list_eps, list_num)
plt.title('Number of selected data')
plt.grid(True)
plt.show()
plt.plot(list_eps, list_per)
plt.title('Percentage of selected data')
plt.grid(True)
plt.show()

plt.plot(list_eps, list_time)
plt.title('Training time')
plt.grid(True)
plt.show()

plt.plot(list_eps, list_train_acc)
plt.title('Accuracy')
plt.grid(True)
plt.show()

plt.plot(list_eps, list_train_loss)
plt.title('Loss')
plt.grid(True)
plt.show()

plt.plot(list_eps, list_test_acc)
plt.title('Accuracy')
plt.grid(True)
plt.show()

plt.plot(list_eps, list_test_loss)
plt.title('Loss')
plt.grid(True)
plt.show()
    







#print the resutls
# print("Initial training evaluation")
# eva_train_ini = model_ini.evaluate(X_train, y_train)
# print("Initial testing evaluation")
# eva_test_ini = model_ini.evaluate(X_test, y_test)
# print("After training evaluation")
# eva_train_slt = model.evaluate(slt_X_train, slt_y_train)
# print("After testing evaluation")
# eva_test_slt = model.evaluate(X_test, y_test)

# print("Optimized training time:",timeDiff_slt)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    