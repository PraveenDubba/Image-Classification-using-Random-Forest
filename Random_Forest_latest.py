# -*- coding: utf-8 -*-
"""
Title: Image Classification using Random Forest

@author: Team I
"""

# setting up the data path
import os 
os.chdir("C:/Users/prave/Downloads/Praveen/UConn/Predictive modeling/My Learnings/Python Project/")

# Importing all the necessary libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split


# Importing Train and Test datasets
train_data = pd.read_csv("datasets/fashion_mnist_train.csv")
final_test_data = pd.read_csv("datasets/fashion_mnist_test.csv")


# Splitting independent variables from the dependent variable in both training and testing
X_train = train_data.iloc[:,1:]
y_train = train_data.label.astype("str")

X_final_test = final_test_data.iloc[:,1:]
y_final_test = final_test_data.label.astype("str")



# Splitting train data into training and validation datasets
x_train, x_test, y_train_v, y_test_v = train_test_split(X_train,y_train, test_size = 0.3, random_state = 2)

# ================== Using Random Forest without hyper paramter tuning and clustering ===================
rf = RandomForestClassifier()

rf.fit(x_train,y_train_v)
# Predictions on training and validation
y_pred_train = rf.predict(x_train)
    # predictions for test
y_pred_test = rf.predict(x_test)
    # training metrics
print("Training metrics:")
print(sklearn.metrics.classification_report(y_true= y_train_v, y_pred= y_pred_train))
    
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test_v, y_pred= y_pred_test))


# Predictions on testset
y_pred_test = rf.predict(X_final_test)
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_final_test, y_pred= y_pred_test))

# Results:
#    86% accuracy on both validation and test datasets


# =========================== Using Grid Search for hyper parameter tuning ===================================
clf = GridSearchCV(rf, param_grid={'n_estimators':[100,200],'min_samples_leaf':[2,3]})
model = clf.fit(x_train,y_train_v)


y_pred_train = model.predict(x_train)
    # predictions for test
y_pred_test = model.predict(x_test)
    # training metrics
print("Training metrics:")
print(sklearn.metrics.classification_report(y_true= y_train_v, y_pred= y_pred_train))
    
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test_v, y_pred= y_pred_test))


# Predictions on testset
y_pred_test = model.predict(X_final_test)
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_final_test, y_pred= y_pred_test))

# ==================== Using Clustering and hyper parameter tuning ============================
# K- means clustering
kmeans = KMeans(n_clusters=10, init='k-means++')

# fitting K means to X_train
kmeans.fit(X_train)
X_train["k_means_label"] = (kmeans.labels_)
X_train["k_means_label"] = X_train["k_means_label"].astype('str')

# Checking column type of K_means_label
X_train["k_means_label"].dtypes
X_train.k_means_label[0:10]
y_train[0:10]

# fitting K means to X_final_test
kmeans.fit(X_final_test)
X_final_test["k_means_label"] = (kmeans.labels_)
X_final_test["k_means_label"] = X_final_test["k_means_label"].astype('str')
y_final_test[0:10]

# Splitting train data into training and validation datasets
x_train, x_test, y_train_v, y_test_v = train_test_split(X_train,y_train, test_size = 0.3, random_state = 2)

# Hyper parameter tuning with new feature
clf = GridSearchCV(rf, param_grid={'n_estimators':[100,200],'min_samples_leaf':[2,3]})
model = clf.fit(x_train,y_train_v)

y_pred_train = model.predict(x_train)
    # predictions for test
y_pred_test = model.predict(x_test)
    # training metrics
print("Training metrics:")
print(sklearn.metrics.classification_report(y_true= y_train_v, y_pred= y_pred_train))
    
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test_v, y_pred= y_pred_test))


# Predictions on testset
y_pred_test = model.predict(X_final_test)
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_final_test, y_pred= y_pred_test))


# =================== Using 5 Fold Cross Validation to check the consistency of the final model ====================
sk_fold = StratifiedKFold(n_splits=5, shuffle=True)

for train_index, test_index in sk_fold.split(x_train, y_train_v):
    train = [x_train.iloc[i,:] for i in train_index]
    y_trn_k = [y_train_v.iloc[i] for i in train_index]
    test = [x_train.iloc[i,:] for i in test_index]
    y_tst_k = [y_train_v.iloc[i] for i in test_index]
    # predictions for train
    model.fit(train, y_trn_k)
    y_pred_train = model.predict(train)
    # predictions for test
    y_pred_test = model.predict(test)
    # training metrics
    print("Training metrics:")
    print(sklearn.metrics.classification_report(y_true= y_trn_k, y_pred= y_pred_train))
    
    # test data metrics
    print("Test data metrics:")
    print(sklearn.metrics.classification_report(y_true= y_tst_k, y_pred= y_pred_test))
    

# predictions on train
y_pred_train = model.predict(X_train)
    # predictions for test
y_pred_test = model.predict(X_final_test)
    # training metrics
print("Training metrics:")
print(sklearn.metrics.classification_report(y_true= y_train, y_pred= y_pred_train))
    
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_final_test, y_pred= y_pred_test))

