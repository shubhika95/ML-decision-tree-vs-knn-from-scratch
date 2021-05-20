#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:09:13 2021

@author: brendonmcguinness
"""
import numpy as np
import pandas as pd
from decision_tree import *
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import log_loss

def confusion_matrix(y, yh):
    n_classes = np.max(y) + 1
    c_matrix = np.zeros((n_classes, n_classes))
    for c1 in range(n_classes):
        for c2 in range(n_classes):
            #(y==c1)*(yh==c2) is 1 when both conditions are true or 0
            c_matrix[c1, c2] = np.sum((y==c1)*(yh==c2))
    return c_matrix

dataset = input('Which data set would you like to run? (breast cancer or hepatitis) ')

if dataset == 'breast cancer':
    
    #import breast cancer data values from different forms of preprocessing
    train_data = np.genfromtxt('clean_bc_data/train_data_cleanedCancer.csv', delimiter=',')
    train_labels = np.genfromtxt('clean_bc_data/train_labels_cleanedCancer.csv', delimiter=',')
    test_data = np.genfromtxt('clean_bc_data/test_data_cleanedCancer.csv', delimiter=',')
    test_labels = np.genfromtxt('clean_bc_data/test_labels_cleanedCancer.csv', delimiter=',')

#    #import breast cancer data values removed
#    train_data = np.genfromtxt('clean_bc_data/train_data_Cancer_cleanRow.csv', delimiter=',')
#    train_labels = np.genfromtxt('clean_bc_data/train_labels_Cancer_cleanRow.csv', delimiter=',')
#    test_data = np.genfromtxt('clean_bc_data/test_data_Cancer_cleanRow.csv', delimiter=',')
#    test_labels = np.genfromtxt('clean_bc_data/test_labels_Cancer_cleanRow.csv', delimiter=',')

elif dataset == 'hepatitis':
    #import hep data
    train_data = np.genfromtxt('clean_hep_data/train_data_cleanedHepatitis.csv', delimiter=',')
    train_labels = np.genfromtxt('clean_hep_data/train_labels_cleanedHepatitis.csv', delimiter=',')
    test_data = np.genfromtxt('clean_hep_data/test_data_cleanedHepatitis.csv', delimiter=',')
    test_labels = np.genfromtxt('clean_hep_data/test_labels_cleanedHepatitis.csv', delimiter=',')
    
    #train_data = np.genfromtxt('clean_hep_data/train_data_hep_cleanRow.csv', delimiter=',')
    #train_labels = np.genfromtxt('clean_hep_data/train_labels_hep_cleanRow.csv', delimiter=',')
    #test_data = np.genfromtxt('clean_hep_data/test_data_hep_cleanRow.csv', delimiter=',')
    #test_labels = np.genfromtxt('clean_hep_data/test_labels_hep_cleanRow.csv', delimiter=',')
    #
else:
    print('Not an option please choose again')
    quit()
    
feat_bool = input('Do you want to cut out lowly correlated features from Appendix figure? (yes or no) ')

#features being removed described by cross correlation analysis in the report
if feat_bool == 'yes' and dataset == 'breast cancer':
    #delete mitosis
    train_data = np.delete(train_data, 7, 1)
    test_data = np.delete(test_data,7,1)
elif feat_bool == 'yes' and dataset == 'hepatitis':
    #delete sgot
    train_data = np.delete(train_data, 15, 1)
    test_data = np.delete(test_data,15,1)
    #delete liver firm
    train_data = np.delete(train_data, 8, 1)
    test_data = np.delete(test_data,8,1)
else:
    pass

#change tree if you want here max depth is 3 and cost misclassification is used                
tree = DecisionTree(max_depth=3,cost_fn=cost_misclassification)
tree = tree.fit(train_data,train_labels.astype(int))
probs_test = tree.predict(test_data)
y_pred = np.argmax(probs_test,1)
probs_train = tree.predict(train_data)
y_train = np.argmax(probs_train,1)
accuracy = tree.evaluate_acc(y_pred, test_labels.astype(int))

cmat = confusion_matrix(test_labels.astype(int), y_pred)

precision = cmat[0,0] / (cmat[0,0]+cmat[0,1])
recall = cmat[0,0] / (cmat[0,0]+cmat[1,0])
print(cmat)
print(f'accuracy: {np.sum(np.diag(cmat))/np.sum(cmat)}')
    
#Plot the error for different max depth values
depth_list = range(1,30)
err_train, err_test = [], []
#loop through differen max depth values
for i, d in enumerate(depth_list):
    #here we're using cost entropy
    tree = DecisionTree(max_depth=d,cost_fn=cost_entropy)
    tree = tree.fit(train_data,train_labels.astype(int))
    probs_test = tree.predict(test_data)
    y_pred = np.argmax(probs_test,1)
    probs_train = tree.predict(train_data)
    y_train = np.argmax(probs_train,1)
    #use misclassification rate I(y != y^) (here its 1-accuracy but scaling factor won't matter)
    err_test.append(1-tree.evaluate_acc(y_pred, test_labels.astype(int)))
    err_train.append(1-tree.evaluate_acc(y_train, train_labels.astype(int)))
    


plt.plot(depth_list, err_test, '-', label='unseen')
plt.plot(depth_list, err_train, '-', label='train')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Misclassification Rate')
plt.title('Loss vs. Max Tree Depth')
plt.show()