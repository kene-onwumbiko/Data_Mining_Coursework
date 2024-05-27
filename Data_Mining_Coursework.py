# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:56:23 2024

@author: keneo
"""

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, SelectFpr
from sklearn.model_selection import cross_val_score, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN

# Get Dataset
dataset = pd.read_csv("Customer-Churn-Records.csv")

# EDA
dataset.info()
head = dataset.head()
tail = dataset.tail()
descriptive = dataset.describe()
correlation_matrix = dataset.corr()
null = dataset.isnull().sum()

# Drop columns
dataset = dataset.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

# Preprocessing data
# Fix balance column - Zero-Inflated Distribution
dataset['Balance_Zero'] = (dataset['Balance'] == 0).astype(int)
dataset['Balance_Log'] = np.log1p(dataset['Balance'])
dataset.loc[dataset['Balance'] == 0, 'Balance_Log'] = -1
dataset = dataset.drop('Balance', axis = 1)

# Drop columns, Dummies, Replace Values
dataset = dataset.drop("Complain", axis = 1)
dataset = pd.get_dummies(dataset, columns = ["Geography", "Gender"], drop_first=True)
dataset = dataset.replace({"SILVER": 0, "GOLD": 1, "PLATINUM": 2, "DIAMOND": 3})

# Select dependent and independent variables
X = dataset.drop("Exited", axis=1)
y = dataset.Exited

# Split dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
# Validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 1/3, random_state = 0, stratify = y_train)

# Resampling minority class
resampler_train = ADASYN(random_state = 0)
X_resampled, y_resampled = resampler_train.fit_resample(X_train, y_train)

# Data Scaling
scaler = MinMaxScaler()
X_resampled[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]] = scaler.fit_transform(X_resampled[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]])
X_train[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]] = scaler.transform(X_train[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]])
X_test[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]] = scaler.transform(X_test[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]])
X_val[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]] = scaler.transform(X_val[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]])

# Feature selection
# ---> Statistical Approach
select_object = SelectFpr(score_func = f_classif)
X_resampled = pd.DataFrame(select_object.fit_transform(X_resampled, y_resampled), columns=select_object.get_feature_names_out())
X_train = pd.DataFrame(select_object.transform(X_train), columns=select_object.get_feature_names_out())
X_test = pd.DataFrame(select_object.transform(X_test), columns=select_object.get_feature_names_out())
X_val = pd.DataFrame(select_object.transform(X_val), columns=select_object.get_feature_names_out())

# Model Training
# CLASSIFIER ONE: Random Forest Classifier
classifier_1 = RandomForestClassifier(max_depth = 5, max_features = None, random_state = 0, warm_start = True)
model_1 = classifier_1.fit(X_resampled, y_resampled)

# Model Prediction
y_pred_1 = model_1.predict(X_resampled)
y_pred_2 = model_1.predict(X_test)
y_pred_3 = model_1.predict(X_val)


# Model Evaluation and Validation
# Training Evaluation
analysis = confusion_matrix(y_resampled, y_pred_1)
class_report = classification_report(y_resampled, y_pred_1)
accuracy = accuracy_score(y_resampled, y_pred_1)
precision = precision_score(y_resampled, y_pred_1)
recall = recall_score(y_resampled, y_pred_1)
f1 = f1_score(y_resampled, y_pred_1)
tpr = analysis[1, 1] / np.sum(analysis[1, :])
tnr = analysis[0, 0] / np.sum(analysis[0, :])
precision_class1 = analysis[1, 1] / np.sum(analysis[:, 1])
precision_class0 = analysis[0, 0] / np.sum(analysis[:, 0])

# Test Evaluations
analysis_1 = confusion_matrix(y_test, y_pred_2)
class_report_1 = classification_report(y_test, y_pred_2)
accuracy_1 = accuracy_score(y_test, y_pred_2)
precision_1 = precision_score(y_test, y_pred_2)
recall_1 = recall_score(y_test, y_pred_2)
f1_score_1 = f1_score(y_test, y_pred_2)
tpr_1 = analysis_1[1, 1] / np.sum(analysis_1[1, :])
tnr_1 = analysis_1[0, 0] / np.sum(analysis_1[0, :])
precision_class1_1 = analysis_1[1, 1] / np.sum(analysis_1[:, 1])
precision_class0_1 = analysis_1[0, 0] / np.sum(analysis_1[:, 0])

# Validation Evaluations
analysis_2 = confusion_matrix(y_val, y_pred_3)
class_report_2 = classification_report(y_val, y_pred_3)
accuracy_2 = accuracy_score(y_val, y_pred_3)
precision_2 = precision_score(y_val, y_pred_3)
recall_2 = recall_score(y_val, y_pred_3)
f1_score_2 = f1_score(y_val, y_pred_3)
tpr_2 = analysis_2[1, 1] / np.sum(analysis_2[1, :])
tnr_2 = analysis_2[0, 0] / np.sum(analysis_2[0, :])
precision_class1_2 = analysis_2[1, 1] / np.sum(analysis_2[:, 1])
precision_class0_2 = analysis_2[0, 0] / np.sum(analysis_2[:, 0])

# Cross Validation
kfold = RepeatedStratifiedKFold(n_splits = 10, random_state = 0, n_repeats = 1)

cross_validation_1 = cross_val_score(model_1, X_train, y_train)
cross_validation_2 = cross_validate(model_1, X_train, y_train, cv = kfold, return_estimator = True, return_train_score = True)

cv_score_mean = round((cross_validation_1.mean() * 100), 2)
cv_score_std = round((cross_validation_1.std() * 100), 2)

fit_time_1 = np.mean(cross_validation_2["fit_time"])
score_time_1 = np.mean(cross_validation_2["score_time"])
test_score_1 = np.mean(cross_validation_2["test_score"])
train_score_1 = np.mean(cross_validation_2["train_score"])




# CLASSIFIER TWO: Gradient Boosting Classifier
classifier_2 = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 300, max_depth = 5, random_state = 0, n_iter_no_change = 10)
model_2 = classifier_2.fit(X_resampled, y_resampled)

# Model Prediction
y_pred_3 = model_2.predict(X_resampled)
y_pred_4 = model_2.predict(X_test)
y_pred_5 = model_2.predict(X_val)


# Model Evaluation and Validation
# Training Evaluation
analysis_3 = confusion_matrix(y_resampled, y_pred_3)
class_report_3 = classification_report(y_resampled, y_pred_3)
accuracy_3 = accuracy_score(y_resampled, y_pred_3)
precision_3 = precision_score(y_resampled, y_pred_3)
recall_3 = recall_score(y_resampled, y_pred_3)
f1_score_3 = f1_score(y_resampled, y_pred_3)
tpr_3 = analysis_3[1, 1] / np.sum(analysis_3[1, :])
tnr_3 = analysis_3[0, 0] / np.sum(analysis_3[0, :])
precision_class1_3 = analysis_3[1, 1] / np.sum(analysis_3[:, 1])
precision_class0_3 = analysis[0, 0] / np.sum(analysis_3[:, 0])

# Test Evaluations
analysis_4 = confusion_matrix(y_test, y_pred_4)
class_report_4 = classification_report(y_test, y_pred_4)
accuracy_4 = accuracy_score(y_test, y_pred_4)
precision_4 = precision_score(y_test, y_pred_4)
recall_4 = recall_score(y_test, y_pred_4)
f1_score_4 = f1_score(y_test, y_pred_4)
tpr_4 = analysis_4[1, 1] / np.sum(analysis_4[1, :])
tnr_4 = analysis_4[0, 0] / np.sum(analysis_4[0, :])
precision_class1_4 = analysis_4[1, 1] / np.sum(analysis_4[:, 1])
precision_class0_4 = analysis_4[0, 0] / np.sum(analysis_4[:, 0])

# Validation Evaluations
analysis_5 = confusion_matrix(y_val, y_pred_5)
class_report_5 = classification_report(y_val, y_pred_5)
accuracy_5 = accuracy_score(y_val, y_pred_5)
precision_5 = precision_score(y_val, y_pred_5)
recall_5 = recall_score(y_val, y_pred_5)
f1_score_5 = f1_score(y_val, y_pred_5)
tpr_5 = analysis_5[1, 1] / np.sum(analysis_5[1, :])
tnr_5 = analysis_5[0, 0] / np.sum(analysis_5[0, :])
precision_class1_5 = analysis_5[1, 1] / np.sum(analysis_5[:, 1])
precision_class0_5 = analysis_5[0, 0] / np.sum(analysis_5[:, 0])

# Cross Validation
kfold = RepeatedStratifiedKFold(n_splits = 10, random_state = 0, n_repeats = 1)

cross_validation_3 = cross_val_score(model_2, X_train, y_train)
cross_validation_4 = cross_validate(model_2, X_train, y_train, cv = kfold, return_estimator = True, return_train_score = True)

cv_score_mean_1 = round((cross_validation_3.mean() * 100), 2)
cv_score_std_1 = round((cross_validation_3.std() * 100), 2)

fit_time_2 = np.mean(cross_validation_4["fit_time"])
score_time_2 = np.mean(cross_validation_4["score_time"])
test_score_2 = np.mean(cross_validation_4["test_score"])
train_score_2 = np.mean(cross_validation_4["train_score"])
