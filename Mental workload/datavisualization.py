# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:58:26 2020

@author: saikumar
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data=pd.read_csv('reg.data114.csv')
y=data[['Hazards','Activity']]
df=data.drop(['id','Hazards','Activity'],axis=1)
scaler=StandardScaler()
scaler.fit(df)
df=scaler.transform(df)
test=np.ones((80,1))
pca=PCA(n_components=2)
pca.fit(df)
rd=pca.transform(df)

plt.figure()
plt.scatter(rd[:,0],rd[:,1],c=y['Activity'])
plt.title('scatter plot for Activity')
plt.xlabel('first principal component')
plt.ylabel('second principal component')

plt.figure()
plt.scatter(rd[:,0],rd[:,1],c=y['Hazards'])
plt.title('scatter plot for Hazards')
plt.xlabel('first principal component')
plt.ylabel('second principal component')

data.boxplot(column='Fixation_frequency',by=['Hazards','Activity'])
data.boxplot(column='Fixation_duration',by=['Hazards','Activity'])
data.boxplot(column='Saccade_duration',by=['Hazards','Activity'])
data.boxplot(column='Saccade_amplitude',by=['Hazards','Activity'])
data.boxplot(column='Fixation_Saccade_ratio',by=['Hazards','Activity'])
data.boxplot(column='MD',by=['Hazards','Activity'])
data.boxplot(column='PD',by=['Hazards','Activity'])
data.boxplot(column='TD',by=['Hazards','Activity'])
data.boxplot(column='P',by=['Hazards','Activity'])
data.boxplot(column='E',by=['Hazards','Activity'])
data.boxplot(column='FR',by=['Hazards','Activity'])
data.boxplot(column='Score',by=['Hazards','Activity'])
data.boxplot(column='RT',by=['Hazards','Activity'])


