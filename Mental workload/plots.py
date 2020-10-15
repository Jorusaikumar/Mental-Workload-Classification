# -*- coding: utf-8 -*-
"""
Created on Sat May 30 07:05:02 2020

@author: saikumar
"""


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
result=pd.read_csv('results.csv')

plt.figure()
plt.ylim(ymin=70,ymax=100)
plt.plot(result.iloc[0:6,0],result.iloc[0:6,1])
plt.plot(result.iloc[6:12,0],result.iloc[6:12,1])
plt.plot(result.iloc[12:18,0],result.iloc[12:18,1])
plt.title('Accuracy vs number of classifiers')
plt.xlabel('number of classifiers')
plt.ylabel('Accuracy')
plt.legend(['1 hidden layer','2 hidden layers','3 hidden layers'])

plt.figure()
plt.ylim(ymin=0,ymax=0.2)
plt.plot(result.iloc[0:6,0],result.iloc[0:6,2])
plt.plot(result.iloc[6:12,0],result.iloc[6:12,2])
plt.plot(result.iloc[12:18,0],result.iloc[12:18,2])
plt.title('Hamming loss vs number of classifiers')
plt.xlabel('number of classifiers')
plt.ylabel('Hamming loss')
plt.legend(['1 hidden layer','2 hidden layers','3 hidden layers'])

plt.figure()
plt.ylim(ymin=0.8,ymax=1)
plt.plot(result.iloc[0:6,0],result.iloc[0:6,3])
plt.plot(result.iloc[6:12,0],result.iloc[6:12,3])
plt.plot(result.iloc[12:18,0],result.iloc[12:18,3])
plt.title('Precision vs number of classifiers')
plt.xlabel('number of classifiers')
plt.ylabel('Precision')
plt.legend(['1 hidden layer','2 hidden layers','3 hidden layers'])

plt.figure()
plt.ylim(ymin=0.8,ymax=1)
plt.plot(result.iloc[0:6,0],result.iloc[0:6,4])
plt.plot(result.iloc[6:12,0],result.iloc[6:12,4])
plt.plot(result.iloc[12:18,0],result.iloc[12:18,4])
plt.title('Recall vs number of classifiers')
plt.xlabel('number of classifiers')
plt.ylabel('Recall')
plt.legend(['1 hidden layer','2 hidden layers','3 hidden layers'])

plt.figure()
plt.ylim(ymin=0.8,ymax=1)
plt.plot(result.iloc[0:6,0],result.iloc[0:6,5])
plt.plot(result.iloc[6:12,0],result.iloc[6:12,5])
plt.plot(result.iloc[12:18,0],result.iloc[12:18,5])
plt.title('F1 score vs number of classifiers')
plt.xlabel('number of classifiers')
plt.ylabel('F1 score')
plt.legend(['1 hidden layer','2 hidden layers','3 hidden layers'])

plt.figure()
plt.ylim(70,100)
for i in range(6):
    l=[]
    l.append(result.iloc[i,1])
    l.append(result.iloc[i+6,1])
    l.append(result.iloc[i+12,1])
    plt.plot(l)
plt.title('Accuracy vs number of hidden layers')
plt.xlabel('Number of hidden layers')
plt.ylabel('Accuracy')
plt.legend(['10','20','30','40','50','60']) 

plt.figure()
plt.ylim(0,0.2)
for i in range(6):
    l=[]
    l.append(result.iloc[i,2])
    l.append(result.iloc[i+6,2])
    l.append(result.iloc[i+12,2])
    plt.plot(l)
plt.title('Hamming loss vs number of hidden layers')
plt.xlabel('Number of hidden layers')
plt.ylabel('Hamming loss')
plt.legend(['10','20','30','40','50','60'])   

plt.figure()
plt.ylim(0.8,1)
for i in range(6):
    l=[]
    l.append(result.iloc[i,3])
    l.append(result.iloc[i+6,3])
    l.append(result.iloc[i+12,3])
    plt.plot(l)
plt.title('precision vs number of hidden layers')
plt.xlabel('Number of hidden layers')
plt.ylabel('precision')
plt.legend(['10','20','30','40','50','60'])      

plt.figure()
plt.ylim(0.8,1)
for i in range(6):
    l=[]
    l.append(result.iloc[i,4])
    l.append(result.iloc[i+6,4])
    l.append(result.iloc[i+12,4])
    plt.plot(l)
plt.title('Recall vs number of hidden layers')
plt.xlabel('Number of hidden layers')
plt.ylabel('Recall')
plt.legend(['10','20','30','40','50','60']) 

plt.figure()
plt.ylim(0.8,1)
for i in range(6):
    l=[]
    l.append(result.iloc[i,5])
    l.append(result.iloc[i+6,5])
    l.append(result.iloc[i+12,5])
    plt.plot(l)
plt.title('F1 score vs number of hidden layers')
plt.xlabel('Number of hidden layers')
plt.ylabel('F1 score')
plt.legend(['10','20','30','40','50','60']) 
