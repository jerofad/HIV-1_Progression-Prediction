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
from sklearn.metrics import classification_report, accuracy_score
#classifiers
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
#Transformation
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
from skopt.space import Real, Integer
from skopt.utils import use_named_args


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

# TODO: Run Hyperparameters tuning on these models.
names = [" Random Forest","Neural Net", "AdaBoost","XGBoost", "Logistic Regression "]
classifiers = [
     RandomForestClassifier(bootstrap=True, max_depth=10, n_estimators=550, criterion="entropy",
                                          max_features='auto', class_weight="balanced", n_jobs=5),
     MLPClassifier(alpha=1,batch_size=30), 
     AdaBoostClassifier(),
     XGBClassifier(),
     LogisticRegression(verbose=5, solver='lbfgs')
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
    # Write the report to a file.
    with open('models_report.txt', 'a') as f:
        print(msg, file=f)
        print('--------------------------------------------------', file=f)
        print(accuracy_score(y_test, predictions), file=f)
        print(classification_report(y_test, predictions), file=f)
        print('--------------------------------------------------', file=f)
        
 ######--------RANDOM FOREST Hyperparameter tuning with forest_minimize-----##### 
 space1  = [Integer(1, 10, name ='max_depth'),
          Integer(10,300, name ='n_estimators'),
          Integer(1, 10, name = 'n_jobs'),
          Integer(2, 10, name = "min_samples_split"),
          Integer(1, 5, name = "min_samples_leaf"),
          Integer(2, 7, name = "max_features")
         ]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective1(**params):
    classifiers[0].set_params(**params)

    return -np.mean(cross_val_score(classifiers[0], X_train,y_train, cv=5, n_jobs=-1,
                                    scoring='roc_auc'))

res_opt = skopt.forest_minimize(objective1, space, n_calls=50, random_state=42)
print("""\t\t Best parameters RANDOM FORESTS:
- max_depth=%d
- n_estimators=%.6f
- n_jobs=%d
- min_samples_split=%d
- min_samples_leaf=%d
- max_features=%d""" % (res_opt.x[0], res_opt.x[1], res_opt.x[2], res_opt.x[3],res_opt.x[4], res_opt.x[5]))

rf = RandomForestClassifier(max_depth=2, n_estimators=274, min_samples_split=6,min_samples_leaf=2, max_features=5, n_jobs=10,random_state = 42)
rf = rf.fit(X_train,y_train)
pred = rf.predict(X_test)
score = sklearn.metrics.accuracy_score(pred, y_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", sklearn.metrics.confusion_matrix(y_test, pred))
print(sklearn.metrics.classification_report(y_test,pred))
 
###--------XGBOOST Hyperparameters tuning with GridSearch-------####
# specify parameters and distributions to sample from
space2 = {"learning_rate": (0.000001, 1.0),#[0.000001, 1.0],
              "max_depth": [2,5],
              "n_estimators": [1, 25],
              "n_jobs":[1,5]
             }
grid_search = GridSearchCV(classifiers[3], space2 ,cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("XGBoost", best_params)
xgboost = XGBClassifier(learning_rate=1e-6, max_depth=2, n_estimators=25, n_jobs=1, random_state=42)
xgboost = xgboost.fit(X_train,y_train)
pred = xgboost.predict(X_test)
score = sklearn.metrics.accuracy_score(pred, y_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", sklearn.metrics.confusion_matrix(y_test, pred))
print(sklearn.metrics.classification_report(y_test,pred))


###--------AdaBoost Hyper-parameter tuning with forest_minimize----######
space3  = [Real(0.000001, 1.0, name='learning_rate'),
          Integer(10, 100,name='n_estimators'),]
@use_named_args(space)
def objective2(**params):
    classifiers[2].set_params(**params)

    return -np.mean(cross_val_score(classifiers[2], X_train,y_train, cv=5, n_jobs=-1,
                                    scoring='roc_auc'))

res_opt = skopt.forest_minimize(objective2, space3, n_calls= 50,random_state=42)

print("""Best parameters ADABOOST:
- n_estimators=%d
- learning rate=%f""" % (res_opt.x[1], res_opt.x[0]))
ada = AdaBoostClassifier(n_estimators=31, learning_rate = 0.304243,random_state= 42)
ada = ada.fit(X_train,y_train)
pred = ada.predict(X_test)
score = sklearn.metrics.accuracy_score(pred, y_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", sklearn.metrics.confusion_matrix(y_test, pred))
print(sklearn.metrics.classification_report(y_test,pred))


####---------MLP Hyper-parameters tuning with forest_minimize------####
space4  = [Real(0.0001, 1.0, name='alpha'),
          Real(1e-4, 1.0, name='tol'),
          Integer(20, 100, name='batch_size'),
          Real(0.1, 0.9, name='momentum'),
          Real(0.000001, 1.0, name='learning_rate_init'),
          Integer(10, 100,name='max_iter'),
          Integer(50, 150, name='hidden_layer_sizes')]

@use_named_args(space)
def objective3(**params):
    classifiers[1].set_params(**params)

    return -np.mean(cross_val_score(classifiers[1], X_train,y_train, cv=5, n_jobs=-1,
                                    scoring='roc_auc'))

res_opt = skopt.forest_minimize(objective3, space4, n_calls= 50,random_state=42)
print("""Best parameters MLP Hyper-parameters:
- alpha=%d
- tol =%f
- batch_size=%d
- momentum=%f
- learning_rate_init=%f
- max_iterations=%d
- hidden layers sizes=%d""" % (res_opt.x[0], res_opt.x[1],res_opt.x[2],res_opt.x[3],res_opt.x[4],res_opt.x[5],res_opt.x[6]))
mlp = MLPClassifier(hidden_layer_sizes= (129,),alpha = 0 ,tol=0.181501, batch_size=89, momentum = 0.873897 ,learning_rate_init= 0.040221 ,max_iter=22,random_state=5)
mlp = mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)
score = sklearn.metrics.accuracy_score(pred, y_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", sklearn.metrics.confusion_matrix(y_test, pred))
print(sklearn.metrics.classification_report(y_test,pred))



####------- Logistic Regression Hyperparameters tuning with forest_minimize--------####
space5  = [Real(1.0, 3.0, name ='C'),
          Integer(10,100, name ='max_iter'),
          Integer(1, 5, name = 'n_jobs'),
          Real(1.0, 5.0, name = "intercept_scaling"),
          Real(1e-4, 1.0, name = "tol"),
          Integer(0, 10, name = "verbose")
         ]
@use_named_args(space)
def objective4(**params):
    classifiers[4].set_params(**params)

    return -np.mean(cross_val_score(classifiers[4], X_train,y_train, cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))

res_opt = skopt.forest_minimize(objective4, space5, n_calls=50, random_state=42)

print("""\t\t Best parameters Logistic Regression:
- C=%d
- max_iter=%.6f
- n_jobs=%d
- intercept_scaling=%d
- tol=%d
- verbose=%d""" % (res_opt.x[0], res_opt.x[1], res_opt.x[2], res_opt.x[3],res_opt.x[4], res_opt.x[5] ))
logreg = LogisticRegression(C=2, max_iter=11, n_jobs=1, random_state=5)
logreg = logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
score = sklearn.metrics.accuracy_score(pred, y_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", sklearn.metrics.confusion_matrix(y_test, pred))
print(sklearn.metrics.classification_report(y_test,pred))


#####-----Stacking all the methods ------####

estimators=[(names[0], rf), 
            (names[1], mlp),
            (names[2], ada),
            (names[3], xgboost),
            (names[4], logreg)
           ]

# Voting based models 
votH_clf = VotingClassifier(estimators, voting='hard').fit(X_train, y_train)
predictions = votH_clf.predict(X_test)
predictions = [round(value) for value in predictions]
with open('models_report.txt', 'a') as f:
    print("Hard Voting Classifier", file=f)
    print('--------------------------------------------------', file=f)
    print(accuracy_score(y_test, predictions), file=f)
    print(classification_report(y_test, predictions), file=f)
    print('--------------------------------------------------', file=f)
