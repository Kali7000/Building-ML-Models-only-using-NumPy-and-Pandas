# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:08:19 2026

@author: Kali
"""

import pandas as pd

# 1. Our mock dataset
data = {
    'Temperature': [65, 85, 55, 30],
    'Season': ['Spring', 'Summer', 'Fall', 'Winter'],
    'Sales': [120, 200, 90, 40]
}
df = pd.DataFrame(data)

print("--- ORIGINAL DATAFRAME ---")
print(df)

# 2. The One-Hot Encoding Fix
# We use the 'drop_first=True' argument to automatically delete the first 
# alphabetical category (in this case, 'Fall') to avoid the Dummy Variable Trap.
df_encoded = pd.get_dummies(df, columns=['Season'], drop_first=True, dtype=float)

print("\n--- ENCODED DATAFRAME (Safe for Regression) ---")
print(df_encoded)