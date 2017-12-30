#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:44:10 2017

@author: fdarmon
"""
import numpy as np
import h5py
from MCB import MCB as MCB_projection

config='test'
norm_mcb =  True
dim_mcb=8000
if config=='train':
    load_projection=None #'dataset_train_tmp.h5'
else :
    if norm_mcb :
        load_projection="dataset_train_MCB_{}_norm.h5".format(dim_mcb)
    else :
        load_projection="dataset_train_MCB_{}.h5".format(dim_mcb)
MCB=(not dim_mcb is None)


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

if not MCB:
    f = h5py.File("dataset_{}.h5".format(config),'w')

    ques_table=f.create_dataset("question",(nb_qu,dim_qu),dtype='float')
    im_table=f.create_dataset("image",(nb_qu,dim_img),dtype='float')

else :
    if norm_mcb :
        filename = "dataset_{}_MCB_{}_norm.h5".format(config,dim_mcb)
    else :
        filename = "dataset_{}_MCB_{}.h5".format(config,dim_mcb)
    f = h5py.File(filename,'w')

    MCB_table=f.create_dataset("data",(nb_qu,dim_mcb),dtype='float')
    # initialize random parameters
    if load_projection is None:
        h_img = np.random.choice(dim_img,dim_mcb)
        s_img = np.random.choice([-1,1],dim_img)
        h_qu = np.random.choice(dim_qu,dim_mcb)
        s_qu = np.random.choice([-1,1],dim_qu)
    else:
        config_f=h5py.File(load_projection)
        h_img = np.array(config_f.get("MCB/images/h"))
        s_img = np.array(config_f.get("MCB/images/s"))
        h_qu = np.array(config_f.get("MCB/questions/h"))
        s_qu = np.array(config_f.get("MCB/questions/s"))
        config_f.close()

    f.create_dataset("MCB/images/h",data=h_img,dtype='int')
    f.create_dataset("MCB/images/s",data=s_img,dtype='int')
    f.create_dataset("MCB/questions/h",data=h_qu,dtype='int')
    f.create_dataset("MCB/questions/s",data=s_qu,dtype='int')

ans_table=f.create_dataset("answer",(nb_qu,),dtype='int')



for i in range(nb_qu):
    if i%1000==0:
        print("Processed {} rows".format(i))
    if MCB:
        img_vector = img_ft[img_corresp[i]-1]
        if norm_mcb :
            img_vector = img_vector / np.sqrt(np.sum(img_vector**2))
        MCB_table[i]=MCB_projection(ques_ft[i],img_vector,dim_mcb,h_qu,s_qu,h_img,s_img)
    else:
        ques_table[i]=ques_ft[i]
        im_table[i]=img_ft[img_corresp[i]-1]

    ans_table[i]=answers[i]

# %%


f.close()
