# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:56:23 2024

@author: keneo
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
# import sweetviz as sv

# Import the dataset
bank_data = pd.read_csv(r"C:\Users\keneo\Downloads\Project Dataset\Customer-Churn-Records.csv")

# # Check the dataset info
# bank_data_info = bank_data.info()

# # Check for missing values
# bank_data_missing = bank_data.isnull().sum()

# # Check for duplicated values
# bank_data_duplicated = bank_data.duplicated().sum()

# # Check the statistics of the dataset
# bank_data_describe = bank_data.describe()

# # Use sweetviz library to perform exploratory data analysis on the dataset
# report = sv.analyze(bank_data)
# report.show_html()

bank_profile = ProfileReport(bank_data, title = "Bank Customer Churn Dataset EDA")
bank_profile.to_file("bank_churn.html")





####################### WITH COMPLAIN COLUMN ########################
# Drop "RowNumber, CustomerId, and Surname"
# Drop "Exited" and reinsert as the last column
new_bank_data = bank_data.drop(columns = ["RowNumber", "CustomerId", "Surname", "Exited"])

# Insert the label at the end of the dataframe
new_bank_data.insert(14, "Exited", bank_data["Exited"])

new_bank_data["Balance"] = new_bank_data["Balance"].replace({0: 10})

new_bank_data["Balance"] = np.log(new_bank_data["Balance"])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Convert "Geography and Gender" to numeric
new_bank_data["Geography"] = label_encoder.fit_transform(new_bank_data["Geography"])
new_bank_data["Gender"] = label_encoder.fit_transform(new_bank_data["Gender"])

# Create a function to replace the values of "Card Type" to numeric with rank
def replace_CardType(a):
    return a.replace({"PLATINUM": 3,
                      "DIAMOND": 2,
                      "GOLD": 1,
                      "SILVER": 0})

# Call the function on the "Card Type"
new_bank_data["Card Type"] = replace_CardType(new_bank_data["Card Type"])

# Separate the features and label 
X = new_bank_data.iloc[:, :-1]
y = new_bank_data.iloc[:, -1]

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, stratify = (y))

# Split the training data into validation and training data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.33, 
                                                              stratify = (y_train))





########## BEFORE BALANCING THE DATA ##########
########## Random Forest Classifier ##########
# Train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Make the prediction
y_pred_rf = rf_model.predict(X_test)

# Get the classification report for the model
class_report_rf = classification_report(y_test, y_pred_rf)

# Get the confusion matrix for the model
plt.rcParams["figure.figsize"] = [15, 10]
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
confusion_matrix_display_rf = ConfusionMatrixDisplay(confusion_matrix_rf, 
                                                     display_labels = rf_model.classes_)
confusion_matrix_display_rf.plot()
plt.title("Confusion Matrix for Randon Forest")
plt.show()

# Get the cross-validation scores for the model
score_rf = {"Accuracy": make_scorer(accuracy_score), 
            "Precision": make_scorer(precision_score, average = "macro"),
            "Recall": make_scorer(recall_score, average = "macro")}
cross_validation_rf = cross_validate(rf_model, X_train, y_train, cv = 5, scoring = score_rf)





########## Gradient Boosting Classifier #########
# Train the model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Make the prediction
y_pred_gb = gb_model.predict(X_test)

# Get the classification report for the model
class_report_gb = classification_report(y_test, y_pred_gb)





########## AFTER BALANCING THE DATA ##########
# Initiate SMOTE Algorithm to balance the data
sm = SMOTE()
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train) 


########## Random Forest Classifier ##########
# Train the model
rf_model_sm = RandomForestClassifier()
rf_model_sm.fit(X_train_sm, y_train_sm)

# Make the prediction
y_pred_rf_sm = rf_model_sm.predict(X_test)

# Get the classification report for the model
class_report_rf_sm = classification_report(y_test, y_pred_rf_sm)


########## Gradient Boosting Classifier #########
# Train the model
gb_model_sm = GradientBoostingClassifier()
gb_model_sm.fit(X_train_sm, y_train_sm)

# Make the prediction
y_pred_gb_sm = gb_model_sm.predict(X_test)

# Get the classification report for the model
class_report_gb_sm = classification_report(y_test, y_pred_gb_sm)





####################### WITHOUT COMPLAIN COLUMN ########################
# Drop "Complain" column
bank_data_without_complain = new_bank_data.drop(columns = "Complain")

# Separate the features and label 
X2 = bank_data_without_complain.iloc[:, :-1]
y2 = bank_data_without_complain.iloc[:, -1]

# Split the data into training and testing data
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.3, stratify = (y2), 
                                                        random_state = 0)





########## BEFORE BALANCING THE DATA ##########
########## Random Forest Classifier ##########
# Train the model
rf_model2 = RandomForestClassifier()
rf_model2.fit(X2_train, y2_train)

# Make the prediction
y2_pred_rf = rf_model2.predict(X2_test)

# Get the classification report for the model
class_report_rf2 = classification_report(y2_test, y2_pred_rf)


########## Gradient Boosting Classifier ##########
# Train the model
gb_model2 = GradientBoostingClassifier()
gb_model2.fit(X2_train, y2_train)

# Make the prediction
y2_pred_gb = gb_model2.predict(X2_test)

# Get the classification report for the model
class_report_gb2 = classification_report(y2_test, y2_pred_gb)





########## AFTER BALANCING THE DATA ##########
# Balance the data
X2_train_sm, y2_train_sm = sm.fit_resample(X2_train, y2_train) 


########## Random Forest Classifier ##########
# Train the model
rf_model2_sm = RandomForestClassifier()
rf_model2_sm.fit(X2_train_sm, y2_train_sm)

# Make the prediction
y2_pred_rf_sm = rf_model2_sm.predict(X2_test)

# Get the classification report for the model
class_report_rf2_sm = classification_report(y2_test, y2_pred_rf_sm)


########## Gradient Boosting Classifier ##########
# Train the model
gb_model2_sm = GradientBoostingClassifier()
gb_model2_sm.fit(X2_train_sm, y2_train_sm)

# Make the prediction
y2_pred_gb_sm = gb_model2_sm.predict(X2_test)

# Get the classification report for the model
class_report_gb2_sm = classification_report(y2_test, y2_pred_gb_sm)



















