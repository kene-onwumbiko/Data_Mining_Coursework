# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:56:23 2024

@author: keneo
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

# Drop features not necessary for building the model
# Drop "RowNumber, Surname, Geography, Gender, Age, Exited"
new_bank_data = bank_data.drop(columns = ["RowNumber", "Surname", "Geography", 
                                          "Gender", "Age", "Exited"])

# Insert the label at the end of the dataframe
new_bank_data.insert(12, "Exited", bank_data["Exited"])

# Separate the features and label 
X = new_bank_data.iloc[:, :12]
y = new_bank_data.iloc[:, 12]

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, stratify = (y), 
                                                    random_state = 0)

# Build a Random Forest Classifier
# Train the model
rf_model = RandomForestClassifier(n_estimators = 100)
rf_model.fit(X_train, y_train)

# Make the prediction
y_pred_rf = rf_model.predict(X_test)












