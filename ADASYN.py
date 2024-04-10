#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import math
from datetime import datetime
import numpy as np
from numpy import argmax

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sys

from imblearn.over_sampling import KMeansSMOTE, ADASYN,SMOTE
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from time import time
from sklearn.metrics import classification_report
from copy import deepcopy
from sklearn.utils import shuffle

# path to the dataset
# file = 'Shaleeza.Dataset.v1.csv'
file = 'IoT Network Intrusion Dataset.csv'
data = pd.read_csv(file)


# In[21]:


def data_preprocessing(targets_others):
    dataset = pd.read_csv(file, error_bad_lines=False, low_memory=False)
    dataset = dataset.drop(['Flow_ID', 'Src_IP', 'Dst_IP', 'Dst_Port', 'Protocol'], axis=1)
    dataset = dataset.drop(['Timestamp'], axis=1)

    # contain only single values
    dataset = dataset.drop(
        ['Fwd_PSH_Flags', 'Fwd_URG_Flags', 'Fwd_Byts/b_Avg', 'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg',
         'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg', 'Init_Fwd_Win_Byts', 'Fwd_Seg_Size_Min'], axis=1)

    dataset['Flow_Byts/s'] = round(dataset['Flow_Byts/s'],2)

    dataset = dataset.drop(targets_others, axis=1)

    dataset = dataset.reset_index()
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)

    # correlation
    correlated_features = set()
    correlation_matrix = dataset.corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= 0.7:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    dataset.drop(labels=correlated_features, axis=1, inplace=True)

    return dataset

def kFoldCV(model, data, n_fold=10):
    diff = int(len(data)/n_fold)
    results = np.zeros((1, 4))
    predictY = deepcopy(data[:,-1])
    for i in range(n_fold):
        begin = diff*i
        end = diff*(i+1)
#         if i == n_fold-1:
#             end = -1
        test = data[begin:end]
        train = deepcopy(data)
        train = np.delete(train, range(begin, end),axis=0)
        X_train, y_train = ADASYN().fit_resample(train[:,:-1], train[:,-1])
        predictY[begin:end] = model.fit(X_train, y_train).predict(test[:,:-1])
    t = classification_report(data[:,-1], predictY)
    print(t)
        
#         results = results + getResults(model, train[:,:-1], train[:,-1],test[:,:-1],test[:,-1])
#     return results/n_fold

def getResults(model, X_train, y_train,X_test,y):
    predictY = model.fit(X_train, y_train).predict(X_test)
    t = classification_report(y, predictY)#, target_names=['0', '1', '2']
    return t


# In[ ]:


# data.groupby('Cat', group_keys=False).apply(lambda x: x.sample(700))
n_samples = 4000
cat = ['DoS','MITM ARP Spoofing','Mirai','Normal','Scan']
data = data_preprocessing(['Label', 'Sub_Cat'])
for c in cat:
    if len(data.loc[data['Cat'] == c]) > n_samples:
        temp = data.loc[data['Cat'] == c].groupby('Cat',group_keys=False).sample(n=n_samples,replace=True,random_state=1024)
        data.drop(data.loc[data['Cat'] == c].index,inplace=True)
        data = pd.concat([data,temp])
data = shuffle(data.values)
X_train, y_train = data[:,:-1], data[:,-1]
y_train = LabelEncoder().fit_transform(y_train)


# In[ ]:


t = time()
length = [3, 4, 5]
for i in length:
    model = DecisionTreeClassifier(max_depth=i)
    kFoldCV(model, data)


print(time()-t)


# In[ ]:


t = time()
hidden_layer_sizes  = [100, 200, 300]
max_iter = [100, 200, 300]
for i in hidden_layer_sizes:
    for j in max_iter:
        model = MLPClassifier(hidden_layer_sizes=i, max_iter=j)
        kFoldCV(model, data)


print(time()-t)


# In[ ]:


t = time()
Cs = [10, 100, 1000]
gammas = [0.01, 0.1,1]
for i in Cs:
    for j in gammas:
        model = SVC(kernel = 'rbf', C = i, gamma = j)
        kFoldCV(model, data)

print(time()-t)


# In[ ]:


n_estimators  = [10, 100, 200]
max_depth = [3, 4, 5]
for i in n_estimators:
    for j in max_depth:
        model = RandomForestClassifier(n_estimators=i, max_depth=j)
        kFoldCV(model, data)

print(time()-t)


# In[ ]:


t = time()
n_estimators  = [10, 100, 200]
max_depth = [3, 4, 5]
for i in n_estimators:
    for j in max_depth:
        model = XGBClassifier(n_estimators=i, max_depth=j,objective='mlogloss')
        kFoldCV(model, data)


print(time()-t)


# In[ ]:


t = time()
n_estimators  = [10, 100, 200]
max_depth = [3, 4, 5]
for i in n_estimators:
    for j in max_depth:
        model = CatBoostClassifier(n_estimators=i, max_depth=j)
        kFoldCV(model, data)

print(time()-t)

