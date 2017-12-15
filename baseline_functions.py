
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

from data_decode import *


############################## KNN BASELINES #######################################


### 1) WITH BOW TFIDF

## compute k nearest questions using tfidf Bag of words
def find_KNN_tdidf(k,idx_test):
    
    #### Parameters #####
    # idx_test : index of the question considered in the test_set 
    # k: parameters of KNN
    
    ### Returns #####
    # idx : indices of the k nearest questions in the training set
    
    cosine_similarities = linear_kernel(tfidf.getrow(n_train+idx_test), tfidf[:n_train])
    idx = np.argsort(cosine_similarities).squeeze()[-k:]
    
    return idx



### KNN baseline as decribed in the article ### 

def predict_answer_KNN_BOW(k,idx_test):
    
    #### Parameters #####
    # idx_test : index of the question considered in the test_set 
    # k: parameters of KNN
    
    ### Returns #####
    # answer : index of the answer predicted in the vocabulary of answers 
    
    knn = find_KNN_tdidf(k,idx_test)
    id_winner = find_closest_im(knn,idx_test)
    idx_answer = answers[id_winner]
    
     
    return idx_answer


                                                             
## find closest images using features images computed by VGG

def find_closest_im(knn,idx_test):
    
    ### Parameters ###
    ## idx_test : index of the question in the test we consider
    # knn : indices of questions in the train 
    
    ## Return ###
    # the index correponding to the closest image to the test image 
    
    im_test =  images_test[img_pos_test[idx_test]]
    
    dist = np.array([scipy.spatial.distance.cosine(im_test,images_train[img_pos_train[k]]) for k in knn])
    
    idx = np.argsort(dist)[0]
    
    return knn[idx]



def evaluate_KNN_BOW(num_test,k):
    
    #### Parameters ####
    # num_test : number of test samples used to evaluate : questions from 0 to num_test-1 are
    # used to test.
    # k : KNN parameter
    
    ### Returns ####
    # test_pred : array of the index of the answer in the vocab for each test sample
    
    test_pred = np.zeros(num_test)
    
    for i in range(num_test):
        test_pred[i]= predict_answer_KNN_BOW(k,i)
    
    return test_pred







#### 2) WITH WORD2VEC



### represent a sentence by the average of the vectors representing each word in the sentence

def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
        
    return feature_vec,n_words




## compute k nearest questions using Word2Vec

def find_knn_word2vec(k,idx_test):
    
    #### Parameters #####
    # idx_test : index of the question considered in the test_set 
    # k: parameters of KNN
    
    ### Returns #####
    # idx : indices of the k nearest questions in the training set
    
    
    sentence = decode_question_test(idx_test,ques_test,vocab)
    
    ## vector representation of the sentence
    
    v =avg_feature_vector(sentence, model, n_features, index2word_set)[0]
    v=(1./lin.norm(v))*v
    
    ## distance with each of the training set sentences
    cosine_similarities = linear_kernel(v.reshape(1,n_features),vectors)
    
    
    knn =  np.argsort(cosine_similarities).squeeze()[-k:]
    
    return knn



def predict_answer_KNN_W2Vec(k,idx_test):
    
    #### Parameters #####
    # idx_test : index of the question considered in the test_set 
    # k: parameters of KNN
    
    ### Returns #####
    # answer : index of the answer predicted in the vocabulary of answers 
    
    knn = find_knn_word2vec(k,idx_test)
    id_winner = find_closest_im(knn,idx_test)
    answer = answers[id_winner]
    
    return answer



def evaluate_KNN_W2V(num_test,k):
    
    #### Parameters ####
    # num_test : number of test samples used to evaluate : questions from 0 to num_test-1 are
    # used to test.
    # k : KNN parameter
    
    ### Returns ####
    # test_pred : array of the index of the answer in the vocab for each test sample
    
    test_pred = np.zeros(num_test)
    
    for i in range(num_test):
        test_pred[i]= predict_answer_KNN_W2Vec(k,i)
    
    return test_pred






# In[2]:


################### READ DATA ########################################################

## read data_prepro.h5

f = h5py.File("data_train_val/data_prepro.h5",'r+')    
keys=list(f.keys())
tmp=[]
for key in keys:
    t=np.array(f.get(key))
    tmp.append(t)

answers = tmp[1]
img_pos_train = tmp[3]-1
img_pos_test = tmp[2]-1
ques_train = tmp[7]
ques_test  = tmp[6]

## number of questions and answers in the training set
n_train = ques_train.shape[0]
n_test = tmp[6].shape[0]


## read data_img.h5

g = h5py.File("data_train_val/data_img.h5",'r+')    
keys=list(g.keys())
tmp_im=[]
for key in keys:
    t=np.array(g.get(key))
    print(key)
    print(t.shape)
    tmp_im.append(t)
    
images_train = tmp_im[1]
images_test = tmp_im[0]


### read test_answers json 
with open('data_train_val/mscoco_val2014_annotations.json') as f:
    ans_test=json.load(f)
    
data = json.load(open('data_train_val/data_prepro.json'))

## vocab of questions
vocab = data['ix_to_word']

## vocab of answers
vocab_ans = data['ix_to_ans']

img= data['unique_img_train']


### decode all the questions training set 
ques_train_decoded=[]

for i in range(n_train):
    ques_train_decoded.append(decode_question(i,ques_train,vocab))
    
### decode questions in the test set 
ques_test_decoded=[]

for i in range(n_test):
    ques_test_decoded.append(decode_question_test(i,ques_test,vocab))


    
### test_true provides for each sampling test, the ground truth and the cartegory of questions

test_true = {}
test_true['ans']=[]
test_true['cat']=[]

for i in range(n_test):
    test_true['ans'].append(ans_test['annotations'][i]['multiple_choice_answer'])
    test_true['cat'].append(ans_test['annotations'][i]['answer_type'])
    
    

################################ EVALUATE #############################################

### Parameters of test ###    

## number of test samples used
num_test= 1000
## KNN parameter
k=4

### KNN BOW ###

### compute tfidf Bag of words  for train + test set
tf_idf = feature_extraction.text.TfidfVectorizer()
tfidf = tf_idf.fit_transform(ques_train_decoded+ques_test_decoded)


test_pred = evaluate_KNN_BOW(num_test,k)
print 'Accuracy results for KNN baseline with BOW tfidf',show_results(test_pred)



### WORD2VEC #### 


import gensim
from gensim.models import Word2Vec

### compute word embedding with word2Vec using train+test questions
### each word is represented by a vector of size n_features=100

questions_tot = ques_train_decoded + ques_test_decoded
Q = [q.split() for q in questions_tot]
model = gensim.models.Word2Vec(Q)
index2word_set = set(model.wv.index2word)

## compute the vector representation of the training sentences
n_features=100
vectors = np.zeros((n_train,n_features))
for i in range(n_train):
    vectors[i,:]= avg_feature_vector(ques_train_decoded[i], model, n_features, index2word_set)[0]
    vectors[i,:]=(1./lin.norm(vectors[i,:]))*vectors[i,:]


test_pred_W2V = evaluate_KNN_W2V(num_test,k)
print 'Accuracy results for KNN baseline with W2vec', show_results(test_pred_W2V)



