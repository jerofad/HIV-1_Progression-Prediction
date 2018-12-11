# -*- coding: utf-8 -*-

"""
  This file is the training file for the final model.
  It trains the model and make predictions. 
  
  Input: trainng data from the get_features.py
  Output: model_report.txt that details the classification report of the model.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#scikit imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#classifiers
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import Matern
#Transformation
from sklearn.preprocessing import StandardScaler
import sklearn
import warnings
warnings.filterwarnings("ignore")
import skopt
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import time
import seaborn as sns
sns.set()

#Clear the text file of previous report.
open('models_report.txt', 'w').close()


#Load train and test datasets
train_Data = pd.read_csv('data/training_new_data.csv')

featureSet = ["VL.t0","CD4.t0","rtlength", "pr_A", "pr_C","pr_G", 
              "pr_R", "pr_T","pr_Y", "PR_GC","RT_A", "RT_C","RT_G","RT_R", "RT_T", "RT_Y", "RT_GC"]
# featureSet = ["VL.t0":"RT_GC"]
X = train_Data[featureSet]
y = train_Data.Resp
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Data transformation with mean 0 and SD 1.
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

# define scoring method
scoring = 'accuracy'

# Define Top 5 models.

names = ["Logistic Regression ","Neural Net", "LDA","GP Classifier", "Gaussian NB"]

# After tuning we have the following parameters. 
# Refer to tuning.py for running with parameters tuning.
classifiers = [
     LogisticRegression(C=2, max_iter=11, n_jobs=1, random_state=5),
     MLPClassifier(random_state = 42,alpha=0, tol =0.031294, batch_size=46, momentum=0.450621,
                   learning_rate_init=0.145296, max_iter=13, hidden_layer_sizes=(86,)),
     LinearDiscriminantAnalysis(n_components=8, tol=0.058093),
     GaussianProcessClassifier(kernel= 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5),
                               n_restarts_optimizer=0,max_iter_predict=39, n_jobs=3),
     GaussianNB(var_smoothing=0.000797)
]
seed = 1
models = zip(names, classifiers)

for name, model in models:
    start = time.clock() 
    kfold = KFold(n_splits=10, random_state = seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print("Running for :", name)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = [round(value) for value in predictions]
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    total_time = (time.clock() - start)
    # Write the report to a file.
    with open('models_report.txt', 'a') as f:
        print(msg, file=f)
        print('--------------------------------------------------', file=f)
        print("\nThe accuracy score that we get is: ", accuracy_score(y_test, predictions), file=f)
        print("\n Confusion Matrix: ", sklearn.metrics.confusion_matrix(y_test, predictions), file=f)
        print(classification_report(y_test, predictions), file=f)
        print("Total time Taken is :",total_time, file=f)
        print('--------------------------------------------------', file=f)



#####-----Stacking all the methods ------####
start = time.clock()
estimators=[(names[0], classifiers[0]), 
            (names[1], classifiers[1]),
            (names[2], classifiers[2]),
            (names[3], classifiers[3]),
            (names[4], classifiers[4])
           ]

# Voting based models 
votH_clf = VotingClassifier(estimators, voting='hard').fit(X_train, y_train)
predictions = votH_clf.predict(X_test)
predictions = [round(value) for value in predictions]
total_time = (time.clock() - start)
with open('models_report.txt', 'a') as f:
    print("Hard Voting Classifier", file=f)
    print('--------------------------------------------------', file=f)
    print(accuracy_score(y_test, predictions), file=f)
    print(classification_report(y_test, predictions), file=f)
    print("\n Total time Taken is :",total_time, file=f)
    print('--------------------------------------------------', file=f)

estimators.append(('Hard Voting Classifier', votH_clf))

# Confusion matrix plot on tets data
f, (ax1) = plt.subplots(2,3)
k = 0
for i in range(2):
    for j in range(3):
        im = ax1[i,j].matshow(confusion_matrix(y_test, estimators[k][1].predict(X_test)),cmap='OrRd')
        ax1[i,j].set(xlabel='Predicted', ylabel='Actual', title = str(estimators[k][0]))
        f.colorbar(im, ax=ax1[i,j])
        k+=1
        
plt.savefig('./Figures/classifies_confusion_matrix.jpg')