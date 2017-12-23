#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:03:56 2017

@author: fdarmon
"""
from keras.utils import Sequence, HDF5Matrix,to_categorical
import math
import numpy as np

class Dataset(Sequence):
    def __init__(self,h5_file,batch_size):
        self.MCB="MCB" in h5_file
        if not self.MCB:
            self.img_h5=h5_file.get("image")
            self.qu_h5=h5_file.get("question")
        else:
            self.mcb_h5=h5_file.get("data")
            self.dim_mcb=self.mcb_h5.shape[1]
        self.ans_h5=h5_file.get("answer")
        self.batch_size=batch_size
        self.size=self.ans_h5.shape[0]


    def __len__(self):
        return(math.ceil(self.size/self.batch_size))

    def __getitem__(self, idx):
        start=idx*self.batch_size
        end=min((idx+1)*self.batch_size,self.size)
        answers=np.array(self.ans_h5[start:end])
        answer_mat=to_categorical(answers-1,num_classes=1000)
        if self.MCB:
            return(self.mcb_h5[start:end],answer_mat)
        else:
            return ([self.qu_h5[start:end],self.img_h5[start:end]],
                    answer_mat)
