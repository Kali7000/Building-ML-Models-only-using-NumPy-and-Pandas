# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:10:58 2026

@author: Kali
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Data
raw_data = pd.read_csv("Ice_cream selling data.csv")
data = raw_data.values.tolist()

# 2. Hyperparameters
w = 0.0 # weight
w2 = 0.0
b = 0.0 # bias
LR = 0.001 # learning rate
epochs = 1000 # Reduced to 1000 for faster standard execution

# 3. Learning Loop (Stochastic Gradient Descent)
for epoch in range(epochs):
    for x, y_actual in data:
        # Predict
        x_sqr = x**2
        #x3 = x**3
        y_predicted = w * x + w2 * x_sqr +  b
        
        # Calculate how far off the guess is
        error = y_predicted - y_actual
        
        # Update w and b to reduce the error
        w = w - (LR * error * x)
        w2 = w2 - (LR * error * x_sqr)
        b = b - (LR * error)
        
    # Print progress every 100 epochs so we don't spam the console
    if epoch % 100 == 0:
        print(f"epoch: {epoch:4d} | w = {w:.2f}, b = {b:.2f} | error = {error}")

print(f"\nFinal parameters -> w = {w:.2f}, b = {b:.2f}")

# --- 4. Visualization ---

# Create a single figure for both the scatter points and the line
plt.figure(figsize=(8, 6))
#usecols=['Temperature (Â°C)', 'Ice Cream Sales (units)']
# Plot the actual scattered data points
plt.scatter(x=raw_data['Temperature'], y=raw_data['Ice Cream Sales (units)'], color='blue', label='Actual Temperature')

# Plot the regression line
# We calculate the predicted line for the entire range of X values
x_values = raw_data['Temperature']
y_values = w * x_values +w2*x_values**2 + b 

plt.plot(x_values, y_values, color='red', linewidth=2, label='Regression Line')

# Format the plot
plt.xlabel('Temperature')
plt.ylabel('Ice Cream Sales ($)')
plt.title('Temperature vs. Ice Cream Sales')
plt.legend()

# Display the final combined plot
plt.show()