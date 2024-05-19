# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:56:23 2024

@author: keneo
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

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
# Drop "RowNumber, CustomerId, Surname, Geography, Gender, Age, Exited"
new_bank_data = bank_data.drop(columns = ["RowNumber", "CustomerId", "Surname", "Geography", 
                                          "Gender", "Age", "Exited"])

# Insert the label at the end of the dataframe
new_bank_data.insert(11, "Exited", bank_data["Exited"])

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = (y), 
                                                    random_state = 0)





########## BEFORE BALANCING THE DATA ##########
########## Random Forest Classifier ##########
# Train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Make the prediction
y_pred_rf = rf_model.predict(X_test)

# Get the classification report for the model
class_report_rf = classification_report(y_test, y_pred_rf)


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



















