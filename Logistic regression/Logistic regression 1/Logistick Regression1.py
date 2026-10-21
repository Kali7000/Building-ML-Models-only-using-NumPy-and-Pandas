# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:44:14 2026

@author: Kali
"""

import math

data = [(1, 0), (2, 0), (4, 1), (5, 1)]
w = 0.0
b = 0.0
lr = 0.1

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# New function just to CALCULATE the Log Loss score
def calculate_log_loss(y_actual, y_pred):
    # Adding a tiny number (1e-15) so we never calculate math.log(0), which crashes Python
    y_pred = max(min(y_pred, 1 - 1e-15), 1e-15) 
    if y_actual == 1:
        return -math.log(y_pred)
    else:
        return -math.log(1 - y_pred)

for epoch in range(1000):
    total_loss = 0 # Let's keep track of our total error
    
    for x, y_actual in data:
        # Step A: Predict
        z = w * x + b
        y_pred = sigmoid(z)
        
        # Monitor: Let's actually calculate the Log Loss to see it!
        total_loss += calculate_log_loss(y_actual, y_pred)
        
        # Step B & C: Update (Using the beautifully simplified derivative)
        error = y_pred - y_actual
        w = w - (lr * error * x)
        b = b - (lr * error)
        
    # Print the loss every 200 epochs to watch it drop!
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Total Log Loss: {total_loss:.4f}")

# (The rest of the testing code is exactly the same...)

# 3. Test the Bouncer
test_hours = 2
probability = sigmoid(w * test_hours + b)
prediction = 1 if probability >= 0.5 else 0

print(f"For {test_hours} hours:")
print(f" - Probability of passing: {probability:.2%}")
print(f" - Final Decision: {'PASS' if prediction == 1 else 'FAIL'}")