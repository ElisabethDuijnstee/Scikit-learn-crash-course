# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 12:53:13 2021

@author: Elisabeth
"""
#https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn#step-3

import pandas as pd
import numpy as np 
import sklearn 
from sklearn.model_selection import train_test_split

#import preprocessing module for ulitities such as scaling, transforming and wrangling data
from sklearn import preprocessing

#import random forest family:
from sklearn.ensemble import RandomForestRegressor

#import tools for cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#import metrics to evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score

#import a way to persist model for future use
#from sklearn.externals import joblib

### get data
data_path = 'C:\\Users\\Elisabeth\\Documents\\Coursera\\Scikit learn\\winequality_red.csv'
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(data_path, sep = ';')
data.head()
data.shape
#describe data in statistics
data.describe()

### splite data into train and test sets 
# y: target features 
y = data.quality
# X: input features
X = data.drop('quality', axis = 1)

# split data into train and test sets using train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 123,stratify = y)
#we set aside 20% of data as a test set
#we set arbitrary 'random state' so we can reproduce results
#stratify to make sure training set looks similar to test test 

### data preprocessing steps:
#1. standardization: substracting means from each feature and 
#divide by feature std dev. 

# =============================================================================
# way of scaling data: 
# X_train_scaled = preprocessing.scale(X_train)
# check: print X_train_scaled.mean(axis=0)
# check: print X_train_scaled.std(axis=0) 
# 
# We will use Transformer API:
# 1. fit transformer on training set (saving means and std dev)
# this is saved in scaler 
# 2. apply transformer on training set (scaling training data)
# 3. apply transformer to test set (using same means and std dev)
# 
# 
# =============================================================================
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))

# in practice we set up the cross-validation pipeline, we dont have to manually fit. 
#we simply declare a modelling pipeline that first transforms the data using StandardScaler
#and then fits a model using a random forst regressor

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = 100))
# modelling pipeline transform using StandardScaler() and than fits model using random forest

### declare hyperparameters to tune. 
#2 types of parameters: (1) model parameters: can be learned from the data (regression coeff etc)
#(2) hyperparameters can not. They express higher level structureal information about the model 
#How many trees, what error to use, max depth of decision tree, etc, should be defined by the user. 

print(pipeline.get_params())

# declare parameters: 
    # add randomforestregressor__ before parameter name when tuned in pipeline
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1]}


# cross validation 
# use CV to evaluate different hyperparameters and estimate effectivenes
# keep test set untained and save for hold out evaluation when you select your model. 
#training model multiple times using the same method 

#1. split data into k equal parts --> preprocess k-1 training folds
#2. train model on k_1 fold (first 9 folds)
#3. evaluate on the remaining hold-out fold (10th) --> preprocessing hold-out fold using same transformations from step (1)
#4. perform steps 2 & 3 k times each time holding out a different fold 
#5. aggregate performance across all k folds (performance metric)

#include data preprocessing steps inside the cross-validation loop 

#sklearn CV with pipeline
#GridSearch CV performs CV across entire grid (all possible permutations) of hyperparameters
clf = GridSearchCV(pipeline, hyperparameters, cv =10)
#fit and tune model:
clf.fit(X_train, y_train)
print(clf.best_params_)

#refit model on entire trainingset. This function is ON by default but we should check:
print(clf.refit)

## evaluate model pipeline on test data

#clf object used to tune hyperparameters can be used directly like a model object. 
#predict new set of data:
y_pred = clf.predict(X_test)

#Use metrics imported earlier to evaluate model performance: 
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#other regression model families: regularized regression, boosted trees
