# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:56:23 2024

@author: keneo
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

import pandas as pd
import sweetviz as sv

# Import the dataset
bank_data = pd.read_csv(r"C:\Users\keneo\Downloads\Project Dataset\Customer-Churn-Records.csv")

# Check the dataset info
bank_data_info = bank_data.info()

# Check for missing values
bank_data_missing = bank_data.isnull().sum()

# Check for duplicated values
bank_data_duplicated = bank_data.duplicated().sum()

# Check the statistics of the dataset
bank_data_describe = bank_data.describe()

# Use sweetviz library to perform exploratory data analysis on the dataset
report = sv.analyze(bank_data)
report.show_html()


####################### WITH COMPLAIN COLUMN ########################
# Drop features not necessary for building the model
# Drop "RowNumber, Surname, Geography, Gender, Age, Exited"
new_bank_data = bank_data.drop(columns = ["RowNumber", "Surname", "Geography", 
                                          "Gender", "Age", "Exited"])

# Insert the label at the end of the dataframe
new_bank_data.insert(12, "Exited", bank_data["Exited"])

# Use LabelEncoder to encode "Card Type"
label_encoder = LabelEncoder()
new_bank_data["Card Type"] = label_encoder.fit_transform(new_bank_data["Card Type"])

# Separate the features and label 
X = new_bank_data.iloc[:, :12]
y = new_bank_data.iloc[:, 12]

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = (y), 
                                                    random_state = 0)

# Build a Random Forest Classifier
# Train the model
rf_model = RandomForestClassifier(n_estimators = 100)
rf_model.fit(X_train, y_train)

# Make the prediction
y_pred_rf = rf_model.predict(X_test)

# Get the classification report for the model
class_report_rf = classification_report(y_test, y_pred_rf)

# Get the cross-validation scores for the model
score_rf = {"Accuracy": make_scorer(accuracy_score), 
            "Precision": make_scorer(precision_score, average = "macro"),
            "Recall": make_scorer(recall_score, average = "macro")}
cross_validation_rf = cross_validate(rf_model, X_train, y_train, cv = 5, scoring = score_rf)


####################### WITHOUT COMPLAIN COLUMN ########################
# Drop "Complain" column
bank_data_without_complain = new_bank_data.drop(columns = "Complain")

# Separate the features and label 
X2 = bank_data_without_complain.iloc[:, :11]
y2 = bank_data_without_complain.iloc[:, 11]

# Split the data into training and testing data
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, stratify = (y2), 
                                                        random_state = 0)

# Build a Random Forest Classifier
# Train the model
rf_model2 = RandomForestClassifier(n_estimators = 100)
rf_model2.fit(X2_train, y2_train)

# Make the prediction
y2_pred_rf = rf_model2.predict(X2_test)

# Get the classification report for the model
class_report_rf2 = classification_report(y2_test, y2_pred_rf)

# Get the cross-validation scores for the model
score_rf2 = {"Accuracy": make_scorer(accuracy_score), 
            "Precision": make_scorer(precision_score, average = "macro"),
            "Recall": make_scorer(recall_score, average = "macro")}
cross_validation_rf2 = cross_validate(rf_model2, X2_train, y2_train, cv = 5, scoring = score_rf2)





