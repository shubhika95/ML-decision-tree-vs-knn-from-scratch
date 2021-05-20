#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:12:40 2021

@author: brendonmcguinness
"""

import numpy as np
import pandas as pd
from decision_tree import *
import matplotlib.pyplot as plt
np.random.seed(1234)

def cross_validate(n, n_folds=10):
    #get the number of data samples in each split
    n_val = n // n_folds
    inds = np.random.permutation(n)
    inds = []
    for f in range(n_folds):
        tr_inds = []
        #get the validation indexes
        val_inds = list(range(f * n_val, (f+1)*n_val))
        #get the train indexes
        if f > 0:
            tr_inds = list(range(f*n_val))
        if f < n_folds - 1:
            tr_inds = tr_inds + list(range((f+1)*n_val, n))
        #The yield statement suspends function’s execution and sends a value back to the caller
        #but retains enoug+h state information to enable function to resume where it is left off
        yield tr_inds, val_inds

def runCrossVall(train_data,train_labels,test_data,test_labels,num_folds=10,depth_list=range(1,30),cost_fun=cost_gini_index):
#num_folds = 10
#depth_list = range(1,30)
    err_test,err_valid = np.zeros(len(depth_list)), np.zeros((len(depth_list),num_folds))
    
    for i, d in enumerate(depth_list):
        #Find the validation errors for num_folds splits for a given K
        for f, (tr, val) in enumerate(cross_validate(train_data.shape[0], num_folds)):
            tree = DecisionTree(max_depth=d,cost_fn=cost_fun)
            tree = tree.fit(train_data[tr],train_labels[tr].astype(int))
            probs_val = tree.predict(train_data[val])
            val_pred = np.argmax(probs_val,1)
            #use misclassification rate I(y != y^) (here its 1-accuracy but scaling factor won't matter)
            err_valid[i,f] = 1-tree.evaluate_acc(train_labels[val], val_pred)
            
            
        #this is the part that we don't do in a usual setup. We don't touch the test set until the very end. 
        tree = DecisionTree(max_depth=d)
        tree = tree.fit(train_data,train_labels.astype(int))
        probs_test = tree.predict(test_data)
        y_pred = np.argmax(probs_test,1)
        #use misclassification rate I(y != y^) (here its 1-accuracy but scaling factor won't matter)
        err_test[i]= 1-tree.evaluate_acc(test_labels,y_pred)
        
    return err_valid, err_test

#train_data = np.genfromtxt('clean_bc_data/train_data_cleanedCancer.csv', delimiter=',')
#train_labels = np.genfromtxt('clean_bc_data/train_labels_cleanedCancer.csv', delimiter=',')
##train_data = train_data[:,1:-1]
#
#test_data = np.genfromtxt('clean_bc_data/test_data_cleanedCancer.csv', delimiter=',')
#test_labels = np.genfromtxt('clean_bc_data/test_labels_cleanedCancer.csv', delimiter=',')
##test_data = test_data[:,1:-1]

#Import data -> Import training and testing data and labels from files you want to run crossval on
#run the preprocessing algorithm first and load the files you want to run the code on
train_data = np.genfromtxt('clean_hep_data/train_data_hep_cleanRow.csv', delimiter=',')
train_labels = np.genfromtxt('clean_hep_data/train_labels_hep_cleanRow.csv', delimiter=',')
test_data = np.genfromtxt('clean_hep_data/test_data_hep_cleanRow.csv', delimiter=',')
test_labels = np.genfromtxt('clean_hep_data/test_labels_hep_cleanRow.csv', delimiter=',')

#running and plotting loss vs tree depth
depth_list = range(1,30)

#call cross validation function for max tree depth range and cost function
err_valid_ent,err_test_ent = runCrossVall(train_data,train_labels,test_data,test_labels,num_folds=10,depth_list=range(1,30),cost_fun=cost_entropy)
err_valid_mis,err_test_mis = runCrossVall(train_data,train_labels,test_data,test_labels,num_folds=10,depth_list=range(1,30),cost_fun=cost_misclassification)
err_valid_gini,err_test_gini = runCrossVall(train_data,train_labels,test_data,test_labels,num_folds=10,depth_list=range(1,30),cost_fun=cost_gini_index)

#plot
plt.plot(depth_list, err_test_ent,  label='test')
plt.errorbar(depth_list, np.mean(err_valid_ent, axis=1), np.std(err_valid_ent, axis=1), label='validation')
plt.legend()
plt.title('10-Fold Cross Validation w/ Cost Entropy (Hepatitis)')
plt.xlabel('Max Tree Depth')
plt.ylabel('Loss')
plt.show()

plt.plot(depth_list, err_test_mis,  label='test')
plt.errorbar(depth_list, np.mean(err_valid_mis, axis=1), np.std(err_valid_mis, axis=1), label='validation')
plt.legend()
plt.title('10-Fold Cross Validation w/ Cost Misclassification (Hepatitis)')
plt.xlabel('Max Tree Depth')
plt.ylabel('Loss')
plt.show()

plt.plot(depth_list, err_test_gini,  label='test')
plt.errorbar(depth_list, np.mean(err_valid_gini, axis=1), np.std(err_valid_gini, axis=1), label='validation')
plt.legend()
plt.title('10-Fold Cross Validation w/ Cost Gini Index (Hepatitis)')
plt.xlabel('Max Tree Depth')
plt.ylabel('Loss')
plt.show()

