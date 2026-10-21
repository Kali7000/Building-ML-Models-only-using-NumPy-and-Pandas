# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:55:27 2026

@author: Kali
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Data
raw_data = pd.read_csv("Salary_dataset.csv", usecols=['YearsExperience', 'Salary'])
data = raw_data.values.tolist()

# 2. Hyperparameters
w = 1.0 # weight
b = 1.0 # bias
LR = 0.001 # learning rate
epochs = 1000 # Reduced to 1000 for faster standard execution

# 3. Learning Loop (Stochastic Gradient Descent)
for epoch in range(epochs):
    for x, y_actual in data:
        # Predict
        y_predicted = w * x + b
        
        # Calculate how far off the guess is
        error = y_predicted - y_actual
        
        # Update w and b to reduce the error
        w = w - (LR * error * x)
        b = b - (LR * error)
        
    # Print progress every 100 epochs so we don't spam the console
    if epoch % 100 == 0:
        print(f"epoch: {epoch:4d} | w = {w:.2f}, b = {b:.2f} | error = {error}")

print(f"\nFinal parameters -> w = {w:.2f}, b = {b:.2f}")

# --- 4. Visualization ---

# Create a single figure for both the scatter points and the line
plt.figure(figsize=(8, 6))

# Plot the actual scattered data points
plt.scatter(x=raw_data['YearsExperience'], y=raw_data['Salary'], color='blue', label='Actual Salary')

# Plot the regression line
# We calculate the predicted line for the entire range of X values
x_values = raw_data['YearsExperience']
y_values = w * x_values + b 

plt.plot(x_values, y_values, color='red', linewidth=2, label='Regression Line')

# Format the plot
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Salary vs. Years of Experience')
plt.legend()

# Display the final combined plot
plt.show()