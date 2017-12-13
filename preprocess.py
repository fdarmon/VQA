#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:44:10 2017

@author: fdarmon
"""
import pandas as pd
import numpy as np
import h5py

features_questions='result/questions_features_train.h5'
questions_file='data_prepro.h5'
features_imgs='data_img.h5'


f=h5py.File(features_questions)
ques_ft_id=np.array(f.get("ques_indexes"))
ques_ft=np.array(f.get("features"))
nb_ques,dim_qu=ques_ft.shape


f.close()

f=h5py.File(questions_file)
img_corresp=np.array(f.get("img_pos_train"))
ques_corresp=np.array(f.get("question_id_train"))
answers=np.array(f.get('answers'))

f.close()
        
f=h5py.File(features_imgs)
img_ft=np.array(f.get("images_train"))
nb_img,dim_img = img_ft.shape

f.close()

final_table=np.zeros((nb_img,dim_qu+dim_img))

# %%
for i in range(nb_img):
    final_table[i,:dim_qu]=ques_ft[i,:]
    ####
    # We assume that ques_f_id and ques_corresp are the same 
    ####
    final_table[i,dim_qu:]=img_ft[img_corresp[i],:] 
        
# %%

f = h5py.File("dataset_train",'w')
f.create_dataset("data",dtype='float',data=final_table)
f.create_dataset("answer",dtype='int',data=answers)
f.close()
        
