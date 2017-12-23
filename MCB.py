
# coding: utf-8
import numpy as np
import scipy.signal


def count_sketch_projection(v,h,s):

    ### Parameters###
    # v : vector to be projected
    # h : indexes random vectors (of size d dimmension of output)
    # s : -1 or 1 random vector (of size d)

    ### Returns ###
    # y : vector projected

    ## input dimension
    n = np.shape(v)[0]
    d=h.shape[0]
    y=np.zeros(d)

    y[h[np.arange(n)]] = y[h[np.arange(n)]] + s*v


    return y


def MCB(v1,v2,d,h1,s1,h2,s2):

    ### Parameters ###
    # v1, v2 : input vectors
    # d : dimension of the resulting vector y

    ### returns ####
    # y : output of the Multi compact bilinear operation applied to vectors v1 and v2
    # MCB consists in the count sketch projection of the outer product of v1 and v2

    v1_proj = count_sketch_projection(v1,h1,s1)
    v2_proj = count_sketch_projection(v2,h2,s2)

    y = scipy.signal.fftconvolve(v1_proj,v2_proj,mode='same')

    return y
