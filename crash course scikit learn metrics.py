# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:40:42 2021

@author: Elisabeth
"""

###Metrics
import pandas as pd
import numpy as np 
import sklearn
import matplotlib.pylab as plt

df = pd.read_csv(r'C:\Users\Elisabeth\Documents\Coursera\Scikit learn\creditcard.csv')[:80_000]
df.head()

"""Dataset has label y that we would like to learn
and X which is data we use to predict label y"""

X = df.drop(columns = ['Time', 'Amount', 'Class']).values
y = df['Class'].values
# X contains all columns that have a v in there
# y is the column that we would like to predict

f"Shapes of X={X.shape} y={y.shape}, #Fraud Cases={y.sum()}"

from sklearn.linear_model import LogisticRegression 
mod = LogisticRegression(class_weight={0:1, 1:2}, max_iter=1000)
#Class_weight tells you how much weight to assign to each class
#class 0 = non fraud, class 1 = fraud (give this double the weight)
mod.fit(X,y).predict(X) #fit on the data and make a prediction
mod.fit(X,y).predict(X).sum()

#now we can move on to grid search and metrics
#use grid search to find the best value for the class_weight
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import precision_score, recall_score, make_scorer
#precision score: given that i predict fraud, how accurate am i?
#revall_score: did i get all the fraud cases?


grid = GridSearchCV(
    estimator = LogisticRegression(max_iter = 1000),
    param_grid = {'class_weight':[{0:1, 1:v} for v in np.linspace(1,20,30)]},
    scoring = {'precision': make_scorer(precision_score), 'recall_score': make_scorer(recall_score)},
    refit = 'precision',
    return_train_score=True,
    cv = 10,
)
grid.fit(X,y)

pd.DataFrame(grid.cv_results_)
recall_score(y, grid.predict(X))


plt.figure(figsize=(12, 4))
df_results = pd.DataFrame(grid.cv_results_)
for score in ['mean_test_recall', 'mean_test_precision', 'mean_test_min_both']:
    plt.plot([_[1] for _ in df_results['param_class_weight']], 
             df_results[score], 
             label=score)
plt.legend();