#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:16:14 2017

@author: fdarmon
"""
from keras.models import Model
from keras.layers import Input, Dense, Multiply, Dropout, Activation
from keras.optimizers import rmsprop
from dataset import Dataset

training_file='dataset_train.h5'
batch_size=32
size=82458

dim_qu=2048
dim_im=4096
split_point=2048

dataset=Dataset(training_file,batch_size,size,split_point)

qu=Input(shape=(dim_qu,),name="question_input")
x_qu = Dense(1024, activation='tanh')(qu)

im=Input(shape=(dim_im,),name="image_input")
x_im = Dense(1024, activation='tanh')(im)

x = Multiply()([x_qu,x_im])
x = Dropout(0.5)(x)
x = Dense(1000,activation='tanh')(x)
x = Dropout(0.5)(x)
x = Dense(1000,activation = 'tanh')(x)
classif = Activation('softmax')(x)
model=Model(inputs=[qu,im],outputs=classif)

opt=rmsprop(lr=5e-4, decay=1e-6)

model.compile(optimizer=opt,loss='categorical_crossentropy')

model.fit_generator(dataset,epochs=2,workers=4,use_multiprocessing=False)



#def perceptron():
