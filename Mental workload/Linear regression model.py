# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:30:28 2020

@author: saikumar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,hamming_loss,precision_score,multilabel_confusion_matrix,recall_score,f1_score

"""===========INPUT DATA AND PREPROCESSING============"""

data=pd.read_csv('reg.data114.csv')
yh=data[['Hazards']]
ya=data[['Activity']]
y=data[['Hazards','Activity']]
X=data.drop(['id','Hazards','Activity'],axis=1)
yh=2-yh
ya=2-ya
y=2-y
X=X.to_numpy()
yh=yh.to_numpy()
ya=ya.to_numpy()
y=y.to_numpy()
X=preprocessing.scale(X)#feature scaling of features

"""=============BOOTSTRAP AGGREGATION====================="""

classifiers=20#no of bootstrap samples to be trained
OOB_list=[[] for _ in range(80)]
Mhlist=[]#list to store hazard models
Malist=[]#list to store activity models
for i in range(classifiers):
    """====BULID THE MODEL===="""
    Hazardmodel=LogisticRegression(max_iter=50)
    Activitymodel=LogisticRegression(max_iter=50)
    
    """====GENERATE A BOOTSTRAP SAMPLE===="""

    m=X.shape[0]
    Xtrain=np.empty(X.shape)
    yhtrain=np.empty(yh.shape)
    yatrain=np.empty(ya.shape)
    all_indices=np.empty(m)
    boot_indices=np.random.randint(0,m,size=m)
    for j in range(m):
        Xtrain[j,:]=X[boot_indices[j],:]
        yhtrain[j,:]=yh[boot_indices[j],:]
        yatrain[j,:]=ya[boot_indices[j],:]
        all_indices[j]=j
      
    """====OUT OF BOX SAMPLE FOR TESTING===="""

    boot_indices=np.unique(boot_indices)
    oob_indices=[int(x) for x in all_indices if x not in boot_indices]
    for itr in range(len(oob_indices)):
        OOB_list[oob_indices[itr]].append(i)
        
    """====TRAINING THE MODEL===="""
    Hazardmodel.fit(Xtrain,yhtrain.reshape(80))
    Activitymodel.fit(Xtrain,yatrain.reshape(80))
    Mhlist.append(Hazardmodel)
    Malist.append(Activitymodel)


"""MAKING PREDICTIONS OF EACH OBSERVATIONS USING CLASSIFIERS IN WHICH IT IS NOT USED FOR TRAINING"""
predictions=np.empty((80,2),dtype='int32')
for i in range(80):
    y_hat=np.array([0,0],dtype='float64')
    for j in range(len(OOB_list[i])):
        predh=Mhlist[OOB_list[i][j]].predict_proba(X[i,:].reshape(1,13))
        preda=Malist[OOB_list[i][j]].predict_proba(X[i,:].reshape(1,13))
        y_hat[0]=y_hat[0]+predh[0,1]
        y_hat[1]=y_hat[1]+preda[0,1]
    y_hat=y_hat/len(OOB_list[i])
    y_hat=(y_hat>=0.5)*1
    predictions[i,:]=y_hat
print("test accuracy:",accuracy_score(predictions,y))
print("hamming loss:",hamming_loss(predictions,y))
print("confusion matrix:")
print(multilabel_confusion_matrix(predictions,y))
print("macro precision:",precision_score(predictions,y,average='macro'))
print("macro recall:",recall_score(predictions,y,average='macro'))
print("macro f1 score:",f1_score(predictions,y,average='macro'))