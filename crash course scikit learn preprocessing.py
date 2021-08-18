# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:40:51 2021

@author: Elisabeth
"""

import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
import sklearn
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures, OneHotEncoder
#standard scaler calculates mean and variance for each column
#quantile transformer is good for outliers
#when data cant be distinguished in just two parts
#change text into numeric 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 

df=pd.read_csv(r"C:\Users\Elisabeth\Documents\Coursera\Scikit learn\datafile_preproc.csv")
df.head()
X = df[['x', 'y']].values
y = df['z'] == "a"

plt.scatter(X[:,0], X[:,1], c=y);

X_new1 = StandardScaler().fit_transform(X)
plt.scatter(X_new1[:,0], X_new1[:,1], c=y);


X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y);
#outliers still there but there effect is much smaller 

#plot_output function to show the difference between model transformers
def plot_output(scaler):
    pipe = Pipeline([
        ("scale", scaler),
        ("model", KNeighborsClassifier(n_neighbors=20, weights='distance'))
    ])

    pred = pipe.fit(X, y).predict(X)

    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Original Data")
    plt.subplot(132)
    X_tfm = scaler.transform(X)
    plt.scatter(X_tfm[:, 0], X_tfm[:, 1], c=y)
    plt.title("Transformed Data")
    plt.subplot(133)
    X_new = np.concatenate([
        np.random.uniform(0, X[:, 0].max(), (5000, 1)), 
        np.random.uniform(0, X[:, 1].max(), (5000, 1))
    ], axis=1)
    y_proba = pipe.predict_proba(X_new)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=y_proba[:, 1], alpha=0.7)
    plt.title("Predicted Data")

plot_output(scaler = StandardScaler())
plot_output(scaler = QuantileTransformer(n_quantiles=100))

df = pd.read_csv(r'C:\Users\Elisabeth\Documents\Coursera\Scikit learn\datafile_preproc_2.csv')
X = df[['x', 'y']].values
y = df['z'] == "a"

plt.scatter(X[:,0], X[:,1], c=y);


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scale", QuantileTransformer(n_quantiles =100)), #quantile transformer as preprocessing step
    ("model", LogisticRegression()) #logistic regression after first prepocessing step 
])

pred = pipe.fit(X, y).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred);

# now lets try a different preprocessing method as this does not work for this data type
pipe = Pipeline([
    ("scale", PolynomialFeatures()), #quantile transformer as preprocessing step
    ("model", LogisticRegression()) #logistic regression after first prepocessing step 
])
pred = pipe.fit(X, y).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred);


###One Hot Encoding
#Make text data numeric data. Take in text and change into something numeric
arr = np.array(["low", "low", "high", "medium"]).reshape(-1, 1)

enc = OneHotEncoder(sparse = False) #sparse is false to see whats inside
enc.fit_transform(arr) #this is the y array to train the model on 
