# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:10:58 2026

@author: Kali
"""

import pandas as pd

# 1. Our Mock Dataset
data = {
    'Applicant': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Education_Level': ['High School', 'PhD', 'Bachelors', 'Masters'],
    'Years_Experience': [2, 10, 4, 7]
}
df = pd.DataFrame(data)

print("--- ORIGINAL DATAFRAME ---")
print(df)

# ==========================================
# 2. THE ORDINAL ENCODING PROCESS
# ==========================================

# Step A: Create a dictionary that explicitly defines the math hierarchy
education_ranking = {
    'High School': 1,
    'Bachelors': 2,
    'Masters': 3,
    'PhD': 4
}

# Step B: Use the Pandas .map() function to swap the text for numbers
df['Education_Encoded'] = df['Education_Level'].map(education_ranking)

# Step C: Drop the original text column so the data is ready for Machine Learning
df = df.drop(columns=['Education_Level'])


print("\n--- ML-READY DATAFRAME ---")
print(df)