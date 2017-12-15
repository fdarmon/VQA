
# coding: utf-8

# In[1]:


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


# In[ ]:


def count_sketch_projection(v,d):

    ### Parameters### 
    # v : vector to be projected 
    # d : dimension output
    
    ### Returns ### 
    # y : vector projected 
    
    ## input dimension
    n = np.shape(v)[0]
    
    h = np.random.choice(d,n)
    s = np.random.choice([-1,1],n)
    
    y=np.zeros(d)
    
    y[h[np.arange(n)]] = y[h[np.arange(n)]] + s*v

    
    return y
    

def MCB(v1,v2,d):
    
    ### Parameters ###
    # v1, v2 : input vectors
    # d : dimension of the resulting vector y
    
    ### returns ####
    # y : output of the Multi compact bilinear operation applied to vectors v1 and v2
    # MCB consists in the count sketch projection of the outer product of v1 and v2
        
    v1_proj = count_sketch_projection(v1,d)
    v2_proj = count_sketch_projection(v2,d)
    
    y = scipy.signal.fftconvolve(v1_proj,v2_proj,mode='same')
    
    return y 
   

