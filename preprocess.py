#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:44:10 2017

@author: fdarmon
"""
import numpy as np
import h5py

config='test'

features_questions='features/questions_features_{}.h5'.format(config)
questions_file='data_prepro.h5'
features_imgs='features/data_img.h5'


f1=h5py.File(features_questions,'r')
ques_ft=f1.get("features")
nb_qu,dim_qu=ques_ft.shape



f2=h5py.File(questions_file,'r')
img_corresp=f2.get("img_pos_{}".format(config))
answers=f2.get('answers')

f3=h5py.File(features_imgs,'r')
img_ft=f3.get("images_{}".format(config))
nb_img,dim_img = img_ft.shape

f = h5py.File("dataset_{}_tmp.h5".format(config),'w')
ques_table=f.create_dataset("question",(nb_qu,dim_qu),dtype='float')
im_table=f.create_dataset("image",(nb_qu,dim_img),dtype='float')
ans_table=f.create_dataset("answer",(nb_qu,),dtype='int')

# %%
for i in range(nb_qu):
    if i%1000==0:
        print("Processed {} rows".format(i))
    ques_table[i]=ques_ft[i]
    ans_table[i]=answers[i]
    im_table[i]=img_ft[img_corresp[i]]

# %%


f.close()
