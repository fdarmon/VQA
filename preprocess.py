#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:44:10 2017

@author: fdarmon
"""
import numpy as np
import h5py

config='train'

features_questions='result/questions_features_{}.h5'.format(config)
questions_file='data_prepro.h5'
features_imgs='data_img.h5'


f1=h5py.File(features_questions,'r')
ques_ft_id=f1.get("ques_indexes")
ques_ft=f1.get("features")
nb_qu,dim_qu=ques_ft.shape



f2=h5py.File(questions_file,'r')
img_corresp=f2.get("img_pos_{}".format(config))
ques_corresp=f2.get("question_id_{}".format(config))
ques_corresp_np=np.array(ques_corresp)
answers=f2.get('answers')

f3=h5py.File(features_imgs,'r')
img_ft=f3.get("images_{}".format(config))
nb_img,dim_img = img_ft.shape

im_table=np.zeros((nb_qu,dim_img))
ques_table=np.zeros((nb_qu,dim_qu))

# %%
for i in range(nb_qu):
    print(i)
    ####
    # We assume that ques_f_id and ques_corresp are the same
    ####
    ques_table[i,:]=ques_ft[i]
    j = np.where(ques_corresp_np==ques_ft_id[i])[0][0]-1 # first indice dans data_prepro qui corresppond à l'identifiant
    img_index=img_corresp[j]
    print(j)
    im_table[i,:]=img_ft[img_index]

# %%

f = h5py.File("dataset_{}".format(config),'w')
f.create_dataset("question",dtype='float',data=ques_table)
f.create_dataset("image",dtype='float',data=im_table)
f.create_dataset("answer",dtype='int',data=answers)
f.close()
