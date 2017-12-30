#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:16:14 2017

@author: fdarmon
"""

from keras.models import Model, load_model
from keras.layers import Input, Dense, Multiply, Dropout, Activation
from keras.optimizers import rmsprop
from keras.callbacks import TensorBoard, LearningRateScheduler
from dataset import Dataset
from norm_layer import Norm_layer
import h5py
import argparse

def main(args):
    training_file=args.input
    batch_size=args.bs
    dim_qu=2048
    dim_im=4096

    f=h5py.File(training_file)
    dataset=Dataset(f,batch_size)

    if args.load == None:
        if dataset.MCB:
            x_in=Input(shape=(dataset.dim_mcb,),name='mcb_input')
            x_mul=x_in
            #x_mul = Norm_layer()(x_in)

        else:
            qu=Input(shape=(dim_qu,),name="question_input")
            x_qu = Dropout(0.5)(qu)
            x_qu = Dense(1024, activation='tanh')(x_qu)

            im=Input(shape=(dim_im,),name="image_input")
            #x_im = Norm_layer()(im)
            x_im = Dropout(0.5)(im)
            x_im = Dense(1024, activation='tanh')(x_im)

            x_mul = Multiply()([x_qu,x_im])
        x = Dropout(0.5)(x_mul)
        x = Dense(1000,activation='tanh')(x)
        x = Dropout(0.5)(x)
        classif = Dense(1000,activation = 'softmax')(x)
        if dataset.MCB:
            inputs=x_in
        else:
            inputs=[qu,im]
        model=Model(inputs=inputs,outputs=classif)

    else :
        model=load_model(args.load,custom_objects={'Norm_layer':Norm_layer})

    opt=rmsprop(lr=args.lr)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    l_callbacks=[]

    if args.decay_every>0:
        l_callbacks.append(LearningRateScheduler(lambda x: args.lr*5*(0.2)**args.decay_every))
    model.fit_generator(dataset,epochs=args.epochs,workers=1,use_multiprocessing=True,callbacks=l_callbacks)

    f.close()
    model.save(args.save)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description = "Training script")
    parser.add_argument("--input",default="./dataset_train.h5",help="Path to training data")
    parser.add_argument("--load", default=None, help='File from which to load model')
    parser.add_argument("--save", default='./trained_model',help="File where to save the model")
    parser.add_argument("--bs",default=500,help='Batch size',type=int)
    parser.add_argument("--epochs",default=1,help="Number of epochs",type=int)
    parser.add_argument("--lr", default=3e-4,help="Learning rate",type=float)
    parser.add_argument("--decay_every",default=-1,help="Learning rate divided by 5 every ...  batch",type=int)

    args=parser.parse_args()
    main(args)
