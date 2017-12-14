#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:03:56 2017

@author: fdarmon
"""
from keras.utils import Sequence, HDF5Matrix,to_categorical
import math
import numpy as np

class Dataset():
    def __init__(self,path,batch_size,size,split_point):
        self.path=path
        self.batch_size=batch_size
        self.size=size
        self.split_point=split_point
        self.answers=np.array(HDF5Matrix(self.path,'answer'))
        self.number_batch_by_buffer=500
        self.size_memory_buffer=self.batch_size*self.number_batch_by_buffer
        self.offset=0
        self.i=0
        self.new_buffer()


    def new_buffer(self):
        print("Updating buffer....\r")
        off=self.offset*self.batch_size
        self.buffer_question=np.array(HDF5Matrix(self.path,'question',off,off+self.size_memory_buffer))
        self.buffer_image=np.array(HDF5Matrix(self.path,'image',off,off+self.size_memory_buffer))
        self.buffer_answer=to_categorical(np.array(HDF5Matrix(self.path,'answer',off,off+self.size_memory_buffer)),num_classes=1000)
        self.offset=self.offset+self.number_batch_by_buffer

    def len(self):
        return(math.ceil(self.size/self.batch_size))

    def getitem(self):

        if self.i-self.offset>=self.number_batch_by_buffer:
            self.new_buffer()
        start=(self.i-self.offset)*self.batch_size
        stop=(self.i+1-self.offset)*self.batch_size

        self.i=self.i+1
        matrix_qu=self.buffer_question[start:stop]
        matrix_im=self.buffer_image[start:stop]
        matrix_ans=self.buffer_answer[start:stop]

        return ([matrix_qu,matrix_im],
                matrix_ans)
