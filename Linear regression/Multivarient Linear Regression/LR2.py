# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:04:18 2026

@author: Kali
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Data
raw_data = pd.read_csv("insurance.csv", usecols=['age','bmi','children','charges'])
#print(raw_data.head())
data = raw_data.values.tolist()

# 2. Hyperparameters
w_age = 1.0 # weight
w_bmi = 1.0
w_children = 1.0
b = 1.0 # bias
LR = 0.000001 # learning rate
epochs = 20000 # Reduced to 1000 for faster standard execution


# 3. Learning Loop (Stochastic Gradient Descent)
for epoch in range(epochs):
    for x1, x2, x3, y_actual in data:
        # Predict
        y_predicted = w_age* x1 + w_bmi*x2 + w_children*x3 + b
        
        # Calculate how far off the guess is
        error = y_predicted - y_actual
        
        # Update w and b to reduce the error
        w_age = w_age - (LR * error * x1)
        w_bmi = w_bmi - (LR * error * x2)
        w_children = w_children - (LR * error * x3)
        b = b - (LR * error)
        
    # Print progress every 100 epochs so we don't spam the console
    if epoch % 100 == 0:
        print(f"epoch: {epoch:4d} | w = {w_age:.2f}, w = {w_children:.2f}, w = {w_bmi:.2f} ; b = {b:.2f}")


print(f"\nFinal parameters -> w = {w_age:.2f}, w = {w_children:.2f}, w = {w_bmi:.2f} ; b = {b:.2f} ")


age = 23
bmi = 24
children = 0

predicted_cost = w_age*age + w_bmi*bmi + w_children*children + b
print(f"predicted_cost: {predicted_cost}")