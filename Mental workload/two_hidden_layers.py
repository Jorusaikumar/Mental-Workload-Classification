# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:29:47 2020

@author: saikumar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,hamming_loss,precision_score,multilabel_confusion_matrix,recall_score,f1_score
from sklearn import preprocessing

"""===========INPUT DATA AND PREPROCESSING============"""

data=pd.read_csv('reg.data114.csv')
y=data[['Hazards','Activity']]
X=data.drop(['id','Hazards','Activity'],axis=1)
y=2-y
X=X.to_numpy()
y=y.to_numpy()
X=preprocessing.scale(X)#feature scaling of features 

"""==============BOOTSTRAP AGGREGATION==========="""

classifiers=10#no of bootstrap samples to be trained
OOB_list=[[] for _ in range(80)]
Mlist=[]#list to store models
tloss=0#training loss
tacc=0#training accuracy

for i in range(classifiers):
    """====BULID THE MODEL===="""
    model=Sequential()
    model.add(Dense(10,activation='sigmoid',input_dim=X.shape[1]))
    model.add(Dense(10,activation='sigmoid'))
    model.add(Dense(2,activation='sigmoid'))
    opt=keras.optimizers.SGD(learning_rate=0.02)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    
    """====GENERATE A BOOTSTRAP SAMPLE===="""
    
    m=X.shape[0]
    Xtrain=np.empty(X.shape)
    ytrain=np.empty(y.shape)
    all_indices=np.empty(m)
    boot_indices=np.random.randint(0,m,size=m)
    for j in range(m):
        Xtrain[j,:]=X[boot_indices[j],:]
        ytrain[j,:]=y[boot_indices[j],:]
        all_indices[j]=j
        
    """====OUT OF BOX SAMPLE FOR TESTING===="""
    
    boot_indices=np.unique(boot_indices)
    oob_indices=[int(x) for x in all_indices if x not in boot_indices]
    for itr in range(len(oob_indices)):
        OOB_list[oob_indices[itr]].append(i)
        
    """====TRAINING THE MODEL===="""
    ep=1500
    history=model.fit(Xtrain,ytrain,epochs=ep,batch_size=10,verbose=0).history
    Mlist.append(model)
    train_loss=history['loss'][ep-1]#last epoch corresponds to training loss
    train_acc=history['accuracy'][ep-1]
    tloss+=train_loss
    tacc+=train_acc

tloss/=classifiers
tacc/=classifiers
"""MAKING PREDICTIONS OF EACH OBSERVATIONS USING CLASSIFIERS IN WHICH IT IS NOT USED FOR TRAINING"""
predictions=np.empty((80,2),dtype='int32')
for i in range(80):
    y_hat=np.array([0,0],dtype='float64')
    for j in range(len(OOB_list[i])):
        pred=Mlist[OOB_list[i][j]].predict(X[i,:].reshape((1,13)))
        y_hat=y_hat+pred
    y_hat=y_hat/len(OOB_list[i])
    y_hat=(y_hat>=0.5)*1
    predictions[i,:]=y_hat
print("training loss and accuracy:",tloss,",",tacc)
print("test accuracy:",accuracy_score(predictions,y))
print("hamming loss:",hamming_loss(predictions,y))
print("confusion matrix:")
print(multilabel_confusion_matrix(predictions,y))
print("macro precision:",precision_score(predictions,y,average='macro'))
print("macro recall:",recall_score(predictions,y,average='macro'))
print("macro f1 score:",f1_score(predictions,y,average='macro'))
