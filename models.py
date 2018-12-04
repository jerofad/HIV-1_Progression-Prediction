# -*- coding: utf-8 -*-

"""
This file lists all models used on the dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#scikit imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report, accuracy_score
#classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import  AdaBoostClassifier
from xgboost import XGBClassifier
# class balancing
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")
#Load train and test datasets
train_Data = pd.read_csv('training_new_data.csv')

featureSet = ["VL.t0","CD4.t0","rtlength", "pr_A", "pr_C","pr_G", 
              "pr_R", "pr_T","pr_Y", "PR_GC","RT_A", "RT_C","RT_G","RT_R", "RT_T", "RT_Y", "RT_GC"]
# featureSet = ["VL.t0":"RT_GC"]
X = train_Data[featureSet]
y = train_Data.Resp
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define scoring method
scoring = 'accuracy'

# Define models 
names = ["Neural Net","SVM ", "AdaBoost","XGBoost", "Logistic Regression"]
# Add more classisifer: Ensemble modelling and Boosting.
classifiers = [
     MLPClassifier(alpha=1,batch_size=30),
     SVC(kernel = 'rbf',), 
     AdaBoostClassifier(),
     XGBClassifier(base_score=1, booster='gbtree',learning_rate=0.1,n_estimators=100),
     LogisticRegression(C=8.0, verbose=5, solver='lbfgs')
]
seed = 1
models = zip(names, classifiers)

# evaluate each model
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state = seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = [round(value) for value in predictions]
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    # TODO: Write this to a file.
    print(msg)
    print('--------------------------------------------------')
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print('--------------------------------------------------')
