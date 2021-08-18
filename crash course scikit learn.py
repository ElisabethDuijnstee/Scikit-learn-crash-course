# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 07:50:16 2021

@author: Elisabeth
"""
"""Scikit learn crash course
1. from sklearn. import different machine learning models
however, the nice thing is that we can stick to the API
but using different machine learning models
We can write one code but change it to KNN, LinReg etc

 """
import sklearn
import pandas as pd
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y = True)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler #can do scaling
from sklearn.pipeline import Pipeline #allows to chain processing steps after each other 
from sklearn.model_selection import GridSearchCV

#mod = LinearRegression()
mod_1 = KNeighborsRegressor()
mod_1.fit(X,y) #model learns from the data
mod_1.predict(X) #model predicts y using X
mod = KNeighborsRegressor().fit(X,y)
pred = mod.predict(X) #gives an array of predictions, and y is an array of orginal values 
plt.scatter(pred,y) #scatterplot to see how well the model is doing, predicted values on x and original values on y


### Building blocks for scikit learn 
#creat pipeline
pipe = Pipeline([ #start pipeline object
    ("scale", StandardScaler()), #() means its an object and not a class. #needs a list of tuples (pair of a name and a step (step = object))
    ("model", KNeighborsRegressor(n_neighbors =1)) #after scaling we want to use our KNN
]) 
pipe.fit(X,y)
pipe.get_params() #every scikitlearn estimator has this. it gives all settings we are able to tweek

#use gridsearch
pred = pipe.predict(X)
plt.scatter(pred,y) #less noise because we scaled

mod = GridSearchCV(estimator = pipe, #estimator has .fit, .predict and pipeline
                   param_grid = {'model__n_neighbors': [1,2,3,4,5,6,7,8,9,10]}, #these are all estimators i would like to checj
                   cv = 3) #we want to let grid search also to do cross validation

mod.fit(X,y);
pd.DataFrame(mod.cv_results_)

### For datascience purposes we want to look at the data set
print(load_boston()['DESCR'])