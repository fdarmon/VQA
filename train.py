#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:16:14 2017

@author: fdarmon
"""
from keras.models import Model
from keras.layers import Input, Dense, Multiply, Dropout, Activation
from keras.optimizers import rmsprop
from keras.callbacks import TensorBoard, LearningRateScheduler
from dataset import Dataset
import h5py

training_file='dataset_train.h5'
batch_size=32
dim_qu=2048
dim_im=4096

f=h5py.File(training_file)
dataset=Dataset(f,batch_size)

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

opt=rmsprop(lr=3e-4)
tb = TensorBoard(log_dir='./logs')
lrs=LearningRateScheduler(lambda x: 3e-4*(0.5)**x)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy','categorical_crossentropy'])

model.fit_generator(dataset,epochs=10,workers=4,use_multiprocessing=True,callbacks=[lrs,])

f.close()
model.save('./trained_model')
