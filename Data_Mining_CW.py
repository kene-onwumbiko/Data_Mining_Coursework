# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:56:23 2024

@author: keneo
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# Import the dataset
bank_data = pd.read_csv(r"C:\Users\keneo\Downloads\Project Dataset\Customer-Churn-Records.csv")

# # Use YData-profiling library to perform Exploratory Data Analysis on the dataset
# bank_profile = ProfileReport(bank_data, title = "Bank Customer Churn Dataset EDA")
# bank_profile.to_file("bank_churn.html")

# # Link to the Exploratory Data Analysis available on Github
# "https://kene-onwumbiko.github.io/Data_Mining_Coursework/bank_churn.html"





####################### WITH COMPLAIN COLUMN ########################
# Drop Irrelevant Columns
# Drop "RowNumber, CustomerId, and Surname"
# Drop "Exited" and reinsert as the last column in the dataset
new_bank_data = bank_data.drop(columns = ["RowNumber", "CustomerId", "Surname", "Exited"])

# Insert "Exited" as the last column in the dataset
new_bank_data.insert(14, "Exited", bank_data["Exited"])


# Handle Zero-inflated Distribution in the Dataset
# Replace the values of "Balance" with the log
new_bank_data["Balance"] = np.log(new_bank_data["Balance"])

# Replace the -inf values with -1
new_bank_data["Balance"] = new_bank_data["Balance"].replace({-np.inf: -1})


# Data Transformation
# Initialise LabelEncoder to encode the categorical values
label_encoder = LabelEncoder()

# Convert "Geography and Gender" to numeric values using LabelEncoder
new_bank_data["Geography"] = label_encoder.fit_transform(new_bank_data["Geography"])
new_bank_data["Gender"] = label_encoder.fit_transform(new_bank_data["Gender"])

# Create a function to replace the values of "Card Type" to numeric
def replace_CardType(a):
    return a.replace({"PLATINUM": 3,
                      "DIAMOND": 2,
                      "GOLD": 1,
                      "SILVER": 0})

# Call the function on "Card Type"
new_bank_data["Card Type"] = replace_CardType(new_bank_data["Card Type"])


# # Split the Dataset
# # Separate the features and label 
# X = new_bank_data.iloc[:, :-1]
# y = new_bank_data.iloc[:, -1]

# # Split the data into training+validation and testing data
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2, stratify = (y))

# # Further split the training+validation data into training and validation data
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.33, 
#                                                   stratify = (y_train_val))

# # Initialise Column Transformer to scale selected columns
# ct_scaler = ColumnTransformer(transformers = [("scale", MinMaxScaler(), 
#                                                [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13])], 
#                               remainder = "passthrough")

# # Fit the Column Transformer on the training data and transform the training data
# X_train_scaled = ct_scaler.fit_transform(X_train)

# # Transform the validation and test data using the fitted Column Transformer
# X_val_scaled = ct_scaler.transform(X_val)
# X_test_scaled = ct_scaler.transform(X_test)





# ########## BEFORE BALANCING THE DATA ##########
# ########## Random Forest Classifier ##########
# # Train the model
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train_scaled, y_train)

# # Make the predictions on the validation data
# y_val_pred_rf = rf_model.predict(X_val_scaled)

# # Get the classification report for the validation data
# class_report_rf_val = classification_report(y_val, y_val_pred_rf)

# # Get the confusion matrix for the validation data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_rf_val = confusion_matrix(y_val, y_val_pred_rf)
# confusion_matrix_display_rf_val = ConfusionMatrixDisplay(confusion_matrix_rf_val, 
#                                                          display_labels = rf_model.classes_)
# confusion_matrix_display_rf_val.plot()
# plt.title("Validation Data Confusion Matrix for Random Forest \n (Before Balancing the Data & Before Dropping COMPLAIN Column)")
# plt.show()

# # Make the predictions on the test data
# y_test_pred_rf = rf_model.predict(X_test_scaled)

# # Get the classification report for the test data
# class_report_rf_test = classification_report(y_test, y_test_pred_rf)

# # Get the confusion matrix for the test data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_rf_test = confusion_matrix(y_test, y_test_pred_rf)
# confusion_matrix_display_rf_test = ConfusionMatrixDisplay(confusion_matrix_rf_test, 
#                                                           display_labels = rf_model.classes_)
# confusion_matrix_display_rf_test.plot()
# plt.title("Test Data Confusion Matrix for Random Forest \n (Before Balancing the Data & Before Dropping COMPLAIN Column)")
# plt.show()

# # Get the cross-validation scores for the model
# score_rf = {"Accuracy": make_scorer(accuracy_score), 
#             "Precision": make_scorer(precision_score, average = "macro"),
#             "Recall": make_scorer(recall_score, average = "macro")}
# cross_validation_rf = cross_validate(rf_model, X_train_scaled, y_train, cv = 5, scoring = score_rf)
# cross_validation_rf = pd.DataFrame(cross_validation_rf)





# ########## Gradient Boosting Classifier #########
# # Train the model
# gb_model = GradientBoostingClassifier()
# gb_model.fit(X_train_scaled, y_train)

# # Make the predictions on the validation data
# y_val_pred_gb = gb_model.predict(X_val_scaled)

# # Get the classification report for the validation data
# class_report_gb_val = classification_report(y_val, y_val_pred_gb)

# # Get the confusion matrix for the validation data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_gb_val = confusion_matrix(y_val, y_val_pred_gb)
# confusion_matrix_display_gb_val = ConfusionMatrixDisplay(confusion_matrix_gb_val, 
#                                                          display_labels = gb_model.classes_)
# confusion_matrix_display_gb_val.plot()
# plt.title("Validation Data Confusion Matrix for Gradient Boosting \n (Before Balancing the Data & Before Dropping COMPLAIN Column)")
# plt.show()

# # Make the predictions on the test data
# y_test_pred_gb = gb_model.predict(X_test_scaled)

# # Get the classification report for the test data
# class_report_gb_test = classification_report(y_test, y_test_pred_gb)

# # Get the confusion matrix for the test data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_gb_test = confusion_matrix(y_test, y_test_pred_gb)
# confusion_matrix_display_gb_test = ConfusionMatrixDisplay(confusion_matrix_gb_test, 
#                                                           display_labels = gb_model.classes_)
# confusion_matrix_display_gb_test.plot()
# plt.title("Test Data Confusion Matrix for Gradient Boosting \n (Before Balancing the Data & Before Dropping COMPLAIN Column)")
# plt.show()

# # Get the cross-validation scores for the model
# score_gb = {"Accuracy": make_scorer(accuracy_score), 
#             "Precision": make_scorer(precision_score, average = "macro"),
#             "Recall": make_scorer(recall_score, average = "macro")}
# cross_validation_gb = cross_validate(gb_model, X_train_scaled, y_train, cv = 5, scoring = score_gb)
# cross_validation_gb = pd.DataFrame(cross_validation_gb)





# ########## AFTER BALANCING THE DATA ##########
# # Handle Class Imbalance
# # Initialise ADASYN Algorithm to balance the data
# adasyn = ADASYN(sampling_strategy = "minority")
# X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train_scaled, y_train) 


# ########## Random Forest Classifier ##########
# # Train the model
# rf_model_balanced = RandomForestClassifier()
# rf_model_balanced.fit(X_train_balanced, y_train_balanced)

# # Make the predictions on the validation data
# y_val_pred_rf_balanced = rf_model_balanced.predict(X_val_scaled)

# # Get the classification report for the validation data
# class_report_rf_val_balanced = classification_report(y_val, y_val_pred_rf_balanced)

# # Get the confusion matrix for the validation data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_rf_val_balanced = confusion_matrix(y_val, y_val_pred_rf_balanced)
# confusion_matrix_display_rf_val_balanced = ConfusionMatrixDisplay(confusion_matrix_rf_val_balanced, 
#                                                                   display_labels = rf_model_balanced.classes_)
# confusion_matrix_display_rf_val_balanced.plot()
# plt.title("Validation Data Confusion Matrix for Random Forest \n (After Balancing the Data & Before Dropping COMPLAIN Column)")
# plt.show()

# # Make the predictions on the test data
# y_test_pred_rf_balanced = rf_model_balanced.predict(X_test_scaled)

# # Get the classification report for the test data
# class_report_rf_test_balanced = classification_report(y_test, y_test_pred_rf_balanced)

# # Get the confusion matrix for the test data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_rf_test_balanced = confusion_matrix(y_test, y_test_pred_rf_balanced)
# confusion_matrix_display_rf_test_balanced = ConfusionMatrixDisplay(confusion_matrix_rf_test_balanced, 
#                                                                    display_labels = rf_model_balanced.classes_)
# confusion_matrix_display_rf_test_balanced.plot()
# plt.title("Test Data Confusion Matrix for Random Forest \n (After Balancing the Data & Before Dropping COMPLAIN Column)")
# plt.show()

# # Get the cross-validation scores for the model
# score_rf_balanced = {"Accuracy": make_scorer(accuracy_score), 
#                      "Precision": make_scorer(precision_score, average = "macro"),
#                      "Recall": make_scorer(recall_score, average = "macro")}
# cross_validation_rf_balanced = cross_validate(rf_model_balanced, X_train_scaled, y_train, cv = 5, 
#                                               scoring = score_rf_balanced)
# cross_validation_rf_balanced = pd.DataFrame(cross_validation_rf_balanced)





# ########## Gradient Boosting Classifier #########
# # Train the model
# gb_model_balanced = GradientBoostingClassifier()
# gb_model_balanced.fit(X_train_balanced, y_train_balanced)

# # Make the predictions on the validation data
# y_val_pred_gb_balanced = gb_model_balanced.predict(X_val_scaled)

# # Get the classification report for the validation data
# class_report_gb_val_balanced = classification_report(y_val, y_val_pred_gb_balanced)

# # Get the confusion matrix for the validation data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_gb_val_balanced = confusion_matrix(y_val, y_val_pred_gb_balanced)
# confusion_matrix_display_gb_val_balanced = ConfusionMatrixDisplay(confusion_matrix_gb_val_balanced, 
#                                                                   display_labels = gb_model_balanced.classes_)
# confusion_matrix_display_gb_val_balanced.plot()
# plt.title("Validation Data Confusion Matrix for Gradient Boosting \n (After Balancing the Data & Before Dropping COMPLAIN Column)")
# plt.show()

# # Make the predictions on the test data
# y_test_pred_gb_balanced = gb_model_balanced.predict(X_test_scaled)

# # Get the classification report for the test data
# class_report_gb_test_balanced = classification_report(y_test, y_test_pred_gb_balanced)

# # Get the confusion matrix for the test data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_gb_test_balanced = confusion_matrix(y_test, y_test_pred_gb_balanced)
# confusion_matrix_display_gb_test_balanced = ConfusionMatrixDisplay(confusion_matrix_gb_test_balanced, 
#                                                                    display_labels = gb_model_balanced.classes_)
# confusion_matrix_display_gb_test_balanced.plot()
# plt.title("Test Data Confusion Matrix for Gradient Boosting \n (After Balancing the Data & Before Dropping COMPLAIN Column)")
# plt.show()

# # Get the cross-validation scores for the model
# score_gb_balanced = {"Accuracy": make_scorer(accuracy_score), 
#                      "Precision": make_scorer(precision_score, average = "macro"),
#                      "Recall": make_scorer(recall_score, average = "macro")}
# cross_validation_gb_balanced = cross_validate(gb_model_balanced, X_train_scaled, y_train, cv = 5, 
#                                               scoring = score_gb_balanced)
# cross_validation_gb_balanced = pd.DataFrame(cross_validation_gb_balanced)





####################### WITHOUT COMPLAIN COLUMN ########################
# Drop "Complain" column
bank_data_without_complain = new_bank_data.drop(columns = "Complain")

# Separate the features and label 
X2 = bank_data_without_complain.iloc[:, :-1]
y2 = bank_data_without_complain.iloc[:, -1]

# Split the data into training+validation and testing data
X2_train_val, X2_test, y2_train_val, y2_test = train_test_split(X2, y2, test_size = 0.2, 
                                                                stratify = (y2))

# Further split the training+validation data into training and validation data
X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train_val, y2_train_val, test_size = 0.33, 
                                                      stratify = (y2_train_val))

# Initialise Column Transformer to scale selected columns
ct_scaler2 = ColumnTransformer(transformers = [("scale", MinMaxScaler(), 
                                                [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12])], 
                               remainder = "passthrough")

# Fit the Column Transformer on the training data and transform the training data
X2_train_scaled = ct_scaler2.fit_transform(X2_train)

# Transform the validation and test data using the fitted Column Transformer
X2_val_scaled = ct_scaler2.transform(X2_val)
X2_test_scaled = ct_scaler2.transform(X2_test)





# ########## BEFORE BALANCING THE DATA ##########
# ########## Random Forest Classifier ##########
# # Train the model
# rf_model2 = RandomForestClassifier()
# rf_model2.fit(X2_train_scaled, y2_train)

# # Make the predictions on the validation data
# y2_val_pred_rf = rf_model2.predict(X2_val_scaled)

# # Get the classification report for the validation data
# class_report_rf2_val = classification_report(y2_val, y2_val_pred_rf)

# # Get the confusion matrix for the validation data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_rf2_val = confusion_matrix(y2_val, y2_val_pred_rf)
# confusion_matrix_display_rf2_val = ConfusionMatrixDisplay(confusion_matrix_rf2_val, 
#                                                           display_labels = rf_model2.classes_)
# confusion_matrix_display_rf2_val.plot()
# plt.title("Validation Data Confusion Matrix for Random Forest \n (Before Balancing the Data & After Dropping COMPLAIN Column)")
# plt.show()

# # Make the predictions on the test data
# y2_test_pred_rf = rf_model2.predict(X2_test_scaled)

# # Get the classification report for the test data
# class_report_rf2_test = classification_report(y2_test, y2_test_pred_rf)

# # Get the confusion matrix for the model
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_rf2_test = confusion_matrix(y2_test, y2_test_pred_rf)
# confusion_matrix_display_rf2_test = ConfusionMatrixDisplay(confusion_matrix_rf2_test, 
#                                                            display_labels = rf_model2.classes_)
# confusion_matrix_display_rf2_test.plot()
# plt.title("Test Data Confusion Matrix for Random Forest \n (Before Balancing the Data & After Dropping COMPLAIN Column)")
# plt.show()

# # Get the cross-validation scores for the model
# score_rf2 = {"Accuracy": make_scorer(accuracy_score), 
#              "Precision": make_scorer(precision_score, average = "macro"),
#              "Recall": make_scorer(recall_score, average = "macro")}
# cross_validation_rf2 = cross_validate(rf_model2, X2_train_scaled, y2_train, cv = 5, scoring = score_rf2)
# cross_validation_rf2 = pd.DataFrame(cross_validation_rf2)





# ########## Gradient Boosting Classifier ##########
# # Train the model
# gb_model2 = GradientBoostingClassifier()
# gb_model2.fit(X2_train_scaled, y2_train)

# # Make the predictions on the validation data
# y2_val_pred_gb = gb_model2.predict(X2_val_scaled)

# # Get the classification report for the validation data
# class_report_gb2_val = classification_report(y2_val, y2_val_pred_gb)

# # Get the confusion matrix for the validation data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_gb2_val = confusion_matrix(y2_val, y2_val_pred_gb)
# confusion_matrix_display_gb2_val = ConfusionMatrixDisplay(confusion_matrix_gb2_val, 
#                                                           display_labels = gb_model2.classes_)
# confusion_matrix_display_gb2_val.plot()
# plt.title("Validation Data Confusion Matrix for Gradient Boosting \n (Before Balancing the Data & After Dropping COMPLAIN Column)")
# plt.show()

# # Make the predictions on the test data
# y2_test_pred_gb = gb_model2.predict(X2_test_scaled)

# # Get the classification report for the test data
# class_report_gb2_test = classification_report(y2_test, y2_test_pred_gb)

# # Get the confusion matrix for the test data
# plt.rcParams["figure.figsize"] = [15, 10]
# confusion_matrix_gb2_test = confusion_matrix(y2_test, y2_test_pred_gb)
# confusion_matrix_display_gb2_test = ConfusionMatrixDisplay(confusion_matrix_gb2_test, 
#                                                            display_labels = gb_model2.classes_)
# confusion_matrix_display_gb2_test.plot()
# plt.title("Test Data Confusion Matrix for Gradient Boosting \n (Before Balancing the Data & After Dropping COMPLAIN Column)")
# plt.show()

# # Get the cross-validation scores for the model
# score_gb2 = {"Accuracy": make_scorer(accuracy_score), 
#              "Precision": make_scorer(precision_score, average = "macro"),
#              "Recall": make_scorer(recall_score, average = "macro")}
# cross_validation_gb2 = cross_validate(gb_model2, X2_train_scaled, y2_train, cv = 5, scoring = score_gb2)
# cross_validation_gb2 = pd.DataFrame(cross_validation_gb2)










######## EXPERIMENT USED FOR THE REPORT ##########
########## AFTER BALANCING THE DATA ##########
# Handle Class Imbalance
# Initialise ADASYN Algorithm to balance the data
adasyn = ADASYN(sampling_strategy = "minority")
X2_train_balanced, y2_train_balanced = adasyn.fit_resample(X2_train_scaled, y2_train) 


########## Random Forest Classifier ##########
# Train the model
rf_model2_balanced = RandomForestClassifier()
rf_model2_balanced.fit(X2_train_balanced, y2_train_balanced)

# Make the predictions on the training data
y2_train_pred_rf_balanced = rf_model2_balanced.predict(X2_train_balanced)

# Get the classification report for the training data
class_report_rf2_train_balanced = classification_report(y2_train_balanced, y2_train_pred_rf_balanced)

# Get the confusion matrix for the training data
plt.rcParams["figure.figsize"] = [15, 10]
confusion_matrix_rf2_train_balanced = confusion_matrix(y2_train_balanced, y2_train_pred_rf_balanced)
confusion_matrix_display_rf2_train_balanced = ConfusionMatrixDisplay(confusion_matrix_rf2_train_balanced, 
                                                                     display_labels = rf_model2_balanced.classes_)
confusion_matrix_display_rf2_train_balanced.plot()
plt.title("Training Data Confusion Matrix for Random Forest \n (After Balancing the Data & After Dropping COMPLAIN Column)")
plt.show()

# Make the predictions on the validation data
y2_val_pred_rf_balanced = rf_model2_balanced.predict(X2_val_scaled)

# Get the classification report for the validation data
class_report_rf2_val_balanced = classification_report(y2_val, y2_val_pred_rf_balanced)

# Get the confusion matrix for the validation data
plt.rcParams["figure.figsize"] = [15, 10]
confusion_matrix_rf2_val_balanced = confusion_matrix(y2_val, y2_val_pred_rf_balanced)
confusion_matrix_display_rf2_val_balanced = ConfusionMatrixDisplay(confusion_matrix_rf2_val_balanced, 
                                                                   display_labels = rf_model2_balanced.classes_)
confusion_matrix_display_rf2_val_balanced.plot()
plt.title("Validation Data Confusion Matrix for Random Forest \n (After Balancing the Data & After Dropping COMPLAIN Column)")
plt.show()

# Make the predictions on the test data
y2_test_pred_rf_balanced = rf_model2_balanced.predict(X2_test_scaled)

# Get the classification report for the test data
class_report_rf2_test_balanced = classification_report(y2_test, y2_test_pred_rf_balanced)

# Get the confusion matrix for the test data
plt.rcParams["figure.figsize"] = [15, 10]
confusion_matrix_rf2_test_balanced = confusion_matrix(y2_test, y2_test_pred_rf_balanced)
confusion_matrix_display_rf2_test_balanced = ConfusionMatrixDisplay(confusion_matrix_rf2_test_balanced, 
                                                                    display_labels = rf_model2_balanced.classes_)
confusion_matrix_display_rf2_test_balanced.plot()
plt.title("Test Data Confusion Matrix for Random Forest \n (After Balancing the Data & After Dropping COMPLAIN Column)")
plt.show()

# Get the cross-validation scores for the model
score_rf2_balanced = {"Accuracy": make_scorer(accuracy_score), 
                      "Precision": make_scorer(precision_score, average = "macro"),
                      "Recall": make_scorer(recall_score, average = "macro")}
cross_validation_rf2_balanced = cross_validate(rf_model2_balanced, X2_train_scaled, y2_train, cv = 5, 
                                               scoring = score_rf2_balanced)
cross_validation_rf2_balanced = pd.DataFrame(cross_validation_rf2_balanced)





########## Gradient Boosting Classifier ##########
# Train the model
gb_model2_balanced = GradientBoostingClassifier()
gb_model2_balanced.fit(X2_train_balanced, y2_train_balanced)

# Make the predictions on the training data
y2_train_pred_gb_balanced = gb_model2_balanced.predict(X2_train_balanced)

# Get the classification report for the training data
class_report_gb2_train_balanced = classification_report(y2_train_balanced, y2_train_pred_gb_balanced)

# Get the confusion matrix for the training data
plt.rcParams["figure.figsize"] = [15, 10]
confusion_matrix_gb2_train_balanced = confusion_matrix(y2_train_balanced, y2_train_pred_gb_balanced)
confusion_matrix_display_gb2_train_balanced = ConfusionMatrixDisplay(confusion_matrix_gb2_train_balanced, 
                                                                     display_labels = gb_model2_balanced.classes_)
confusion_matrix_display_gb2_train_balanced.plot()
plt.title("Training Data Confusion Matrix for Gradient Boosting \n (After Balancing the Data & After Dropping COMPLAIN Column)")
plt.show()

# Make the predictions on the validation data
y2_val_pred_gb_balanced = gb_model2_balanced.predict(X2_val_scaled)

# Get the classification report for the validation data
class_report_gb2_val_balanced = classification_report(y2_val, y2_val_pred_gb_balanced)

# Get the confusion matrix for the validation data
plt.rcParams["figure.figsize"] = [15, 10]
confusion_matrix_gb2_val_balanced = confusion_matrix(y2_val, y2_val_pred_gb_balanced)
confusion_matrix_display_gb2_val_balanced = ConfusionMatrixDisplay(confusion_matrix_gb2_val_balanced, 
                                                                   display_labels = gb_model2_balanced.classes_)
confusion_matrix_display_gb2_val_balanced.plot()
plt.title("Validation Data Confusion Matrix for Gradient Boosting \n (After Balancing the Data & After Dropping COMPLAIN Column)")
plt.show()

# Make the predictions on the test data
y2_test_pred_gb_balanced = gb_model2_balanced.predict(X2_test_scaled)

# Get the classification report for the test data
class_report_gb2_test_balanced = classification_report(y2_test, y2_test_pred_gb_balanced)

# Get the confusion matrix for the test data
plt.rcParams["figure.figsize"] = [15, 10]
confusion_matrix_gb2_test_balanced = confusion_matrix(y2_test, y2_test_pred_gb_balanced)
confusion_matrix_display_gb2_test_balanced = ConfusionMatrixDisplay(confusion_matrix_gb2_test_balanced, 
                                                                    display_labels = gb_model2_balanced.classes_)
confusion_matrix_display_gb2_test_balanced.plot()
plt.title("Test Data Confusion Matrix for Gradient Boosting \n (After Balancing the Data & After Dropping COMPLAIN Column)")
plt.show()

# Get the cross-validation scores for the model
score_gb2_balanced = {"Accuracy": make_scorer(accuracy_score), 
                      "Precision": make_scorer(precision_score, average = "macro"),
                      "Recall": make_scorer(recall_score, average = "macro")}
cross_validation_gb2_balanced = cross_validate(gb_model2_balanced, X2_train_scaled, y2_train, cv = 5, 
                                               scoring = score_gb2_balanced)
cross_validation_gb2_balanced = pd.DataFrame(cross_validation_gb2_balanced)


















