# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:56:23 2024

@author: keneo
"""

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




















