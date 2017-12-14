#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:16:14 2017

@author: fdarmon
"""
from keras.models import Model
from keras.layers import Input, Dense, Multiply, Dropout, Activation
from keras.optimizers import adam
from dataset import Dataset
import h5py

training_file='dataset_train.h5'
batch_size=32
with h5py.File(training_file,'r') as f:
    size,dim_qu=f.get("question").shape
    dim_im = f.get("image").shape[1]

dataset=Dataset(training_file,batch_size,size)

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

opt=adam()

model.compile(optimizer=opt,loss='categorical_crossentropy')

model.fit_generator(dataset,epochs=2,workers=4,use_multiprocessing=False)

model.save('./trained_model')
