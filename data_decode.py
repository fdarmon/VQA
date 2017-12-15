
# coding: utf-8

# In[6]:


import h5py
import numpy as np
import json
from sklearn import feature_extraction
from numpy import linalg as lin
from sklearn.metrics.pairwise import linear_kernel
import pyemd
import scipy
import time
get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




## decode answer in the training set
def decode_answer(idx,vocab_ans,answers):
    
    ### Parameter ###
    # idx: index of the answer to decode in the train set
    
    ### Returns ###
    # answer : the content of the corresponding anwser in the train set
    
    return vocab_ans['%d'%answers[idx]]



def decode_answer_test(idx,ans_test):
    
    ### Parameters ###
    # idx : index of the questions in the test set
    
    ### Returns ###
    # the decoded string correponding to the answer in the test set.
    
    return ans_test['annotations'][idx]['multiple_choice_answer']
    

    
#### decode question in the training set 
def decode_question(idx,ques_train,vocab):
    
    ### Parameter ###
    # idx: index of the question to decode
    
    ### Returns ###
    # questions : the content of the corresponding question
    
    ## list of words contained in the question
    word_list = []
    
    for i in ques_train[idx,:]:  #[:tmp[5][idx]]
        if (i!=0):
            word_list.append(vocab['%d'%i])
        
    question =' '.join(word for word in word_list) 
    
    return question


def decode_question_test(idx,ques_test,vocab):
    
    ### Parameter ###
    # idx: index of the question to decode in the test set
    
    ### Returns ###
    # questions : the content of the corresponding question
    
    ## list of words contained in the question
    word_list = []
    
    for i in ques_test[idx,:]:  
        if (i!=0):
            word_list.append(vocab['%d'%i])
        
    question =' '.join(word for word in word_list) 
    
    return question


#### show image corresponding to the questions of index idx in the training set
def show_image(idx,data,img_pos_train):
    
    pos = img_pos_train[idx]
    img = mpimg.imread(data['unique_img_train'][pos])
    imgplot = plt.imshow(img)    
    

#### show image corresponding to the questions of index idx in the test set
def show_image_test(idx,data,img_pos_test):
    
    pos = img_pos_test[idx]
    img = mpimg.imread(data['unique_img_test'][pos])
    imgplot = plt.imshow(img)  

    

