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
    def __init__(self,path,batch_size,size,split_point):
        self.path=path
        self.batch_size=batch_size
        self.size=size
        self.split_point=split_point
        
    def __len__(self):
        return(math.ceil(self.size/self.batch_size))
        
    def __getitem__(self, idx):
        start=idx*self.batch_size
        end=min((idx+1)*self.batch_size,self.size)
        matrix=np.array(HDF5Matrix(self.path,'data',start,end))
        answer_mat=to_categorical(np.array(HDF5Matrix(self.path,'answer',start,end)),num_classes=1000)
        return ([matrix[:,:self.split_point],matrix[:,self.split_point:]],
                answer_mat)