#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:27:50 2021

@author: brendonmcguinness
"""

import numpy as np
import pandas as pd
from decision_tree import *
import matplotlib.pyplot as plt



#import breast cancer data change to where ever your files generated from the preprocessing step are 
#here we're doing decision boundary plots for both datasets 
train_data_bc = np.genfromtxt('clean_bc_data/train_data_cleanedCancer.csv', delimiter=',')
train_labels_bc = np.genfromtxt('clean_bc_data/train_labels_cleanedCancer.csv', delimiter=',')
test_data_bc = np.genfromtxt('clean_bc_data/test_data_cleanedCancer.csv', delimiter=',')
test_labels_bc = np.genfromtxt('clean_bc_data/test_labels_cleanedCancer.csv', delimiter=',')
#import hepatitis data
train_data_hep = np.genfromtxt('clean_hep_data/train_data_hep_cleanRow.csv', delimiter=',')
train_labels_hep = np.genfromtxt('clean_hep_data/train_labels_hep_cleanRow.csv', delimiter=',')
test_data_hep = np.genfromtxt('clean_hep_data/test_data_hep_cleanRow.csv', delimiter=',')
test_labels_hep = np.genfromtxt('clean_hep_data/test_labels_hep_cleanRow.csv', delimiter=',')

#combine all data so that we can get meshgrid
data_bc = np.vstack((train_data_bc,test_data_bc))
data_labels_bc = np.concatenate((train_labels_bc,test_labels_bc))
data_labels_bc = data_labels_bc.astype(int)

data_hep = np.vstack((train_data_hep,test_data_hep))
data_labels_hep = np.concatenate((train_labels_hep,test_labels_hep))
data_labels_hep = data_labels_hep.astype(int)

#we're interested in  bilirubin (13) and albumin (16) for hepatitis
data_red_hep = np.column_stack((data_hep[:,13],data_hep[:,16]))
#and uniformity of cell size and shape for breast cancer data 
data_red_bc = data_bc[:,1:2]

#make the grids
x0v = np.linspace(np.min(data_bc[:,0]), np.max(data_bc[:,0]), 200)
x1v = np.linspace(np.min(data_bc[:,1]), np.max(data_bc[:,1]), 200)
x0,x1 = np.meshgrid(x0v, x1v)
x_all = np.vstack((x0.ravel(),x1.ravel())).T

x0v_h = np.linspace(np.min(data_hep[:,13]), np.max(data_hep[:,13]), 200)
x1v_h = np.linspace(np.min(data_hep[:,16]), np.max(data_hep[:,16]), 200)
x0_h,x1_h = np.meshgrid(x0v_h, x1v_h)
x_all_h = np.vstack((x0_h.ravel(),x1_h.ravel())).T

#here showing max tree depth of 5 just cause it's a little more interesting to see where the additional partitions are made
model = DecisionTree(max_depth=5)

#plotting breast cancer decision boundaries
y_prob_all = model.fit(data_red_bc, data_labels_bc).predict(x_all.astype(int))
y_pred_all = np.argmax(y_prob_all,axis=-1)
plt.scatter(data_bc[:,1], data_bc[:,2], c=data_labels_bc, marker='o', alpha=1)
plt.scatter(x_all[:,0], x_all[:,1], c=y_pred_all, marker='.', alpha=.01)
plt.ylabel('Uniformity of cell size')
plt.xlabel('Uniformity of cell shape')
plt.title('Decision Tree boundries for selected features (Breast Cancer)')
plt.show()

#plotting hepatitis decision boundaries
y_prob_all_h = model.fit(data_red_hep, data_labels_hep).predict(x_all_h.astype(int))
y_pred_all_h = np.argmax(y_prob_all_h,axis=-1)
plt.scatter(data_hep[:,13], data_hep[:,16], c=data_labels_hep, marker='o', alpha=1)
plt.scatter(x_all_h[:,0], x_all_h[:,1], c=y_pred_all_h, marker='.', alpha=.01)
plt.ylabel('Bilirubin')
plt.xlabel('Albumin')
plt.title('Decision Tree boundries for selected features (Hepatitis)')
plt.show()
