# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:10:44 2026

@author: Kali
"""
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
algea_df = pd.read_csv(
    'algeas.csv',
    usecols=['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2', 'Population']
)

print(algea_df.head())

# ----------------------------
# Standardize all columns
# ----------------------------
cols = ['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2', 'Population']

means = algea_df[cols].mean()
stds = algea_df[cols].std()

for col in cols:
    algea_df[col] = (algea_df[col] - means[col]) / stds[col]

print(algea_df.head())

# ----------------------------
# Train/test split
# ----------------------------
algea_df = algea_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(algea_df.shape)

train_rows = int(0.8 * algea_df.shape[0])
train_set = algea_df[:train_rows]
test_set = algea_df[train_rows:]

print(train_set.shape)
print(test_set.shape)

train_data = train_set.values.tolist()
test_data = test_set.values.tolist()





# ----------------------------
# Initialize weights
# 1 bias + 7 linear + 7 squared + 21 interactions = 36 weights
# ----------------------------
weights = [0.0] * 36 

# ----------------------------
# Hyperparameters
# ----------------------------
learning_rate = 0.02
epochs = 2000
#ridge lambda
lambda_val = 0.5

# ----------------------------
# Build feature vector
# Order:
# [1,
#  x1..x7,
#  x1^2..x7^2,
#  x1x2, x1x3, ..., x6x7]
# ----------------------------
def build_features(x1, x2, x3, x4, x5, x6, x7):
    x = [x1, x2, x3, x4, x5, x6, x7]
    features = [1.0]  # bias

    # Linear terms
    for xi in x:
        features.append(xi)

    # Squared terms
    for xi in x:
        features.append(xi ** 2)

    # Interaction terms
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            features.append(x[i] * x[j])

    return features

# ----------------------------
# Prediction
# ----------------------------
def predict(x1, x2, x3, x4, x5, x6, x7):
    features = build_features(x1, x2, x3, x4, x5, x6, x7)
    
    y_pred = 0.0 # Start at 0
    for w, f in zip(weights, features):
        y_pred += w * f  # Accumulate the total score!
        
    return y_pred

# ----------------------------
# MSE
# ----------------------------
def mse(data):
    total = 0.0
    for x1, x2, x3, x4, x5, x6, x7, y_actual in data:
        y_pred = predict(x1, x2, x3, x4, x5, x6, x7)
        total += (y_pred - y_actual) ** 2
    return total / len(data)

# ----------------------------
# Convert prediction back to original scale
# ----------------------------
def predict_original_scale(x1, x2, x3, x4, x5, x6, x7):
    y_pred_scaled = predict(x1, x2, x3, x4, x5, x6, x7)
    return y_pred_scaled * stds["Population"] + means["Population"]

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(epochs):
    for x1, x2, x3, x4, x5, x6, x7, y_actual in train_data:      
        features = build_features(x1, x2, x3, x4, x5, x6, x7)
        
        y_pred = 0.0 # Start at 0
        for w, f in zip(weights, features):
            y_pred += w * f  # Accumulate the total score!
            
        error = y_pred - y_actual

        # Update all weights using a loop
        for i in range(len(weights)):
            if i == 0:
                # The Bias (index 0): NO PENALTY
                weights[i] -= learning_rate * (error * features[i])
            else:  
                weights[i] -= (learning_rate * error * features[i] + lambda_val*weights[i])

    if epoch % 100 == 0:
        train_loss = mse(train_data)
        test_loss = mse(test_data)
        print(f"Epoch {epoch}: train MSE = {train_loss:.4f}, test MSE = {test_loss:.4f}")
        


# ----------------------------
# Final results
# ----------------------------
print("\nFinal model parameters:")
for i, w in enumerate(weights):
    print(f"w{i} = {w}")

print("\nTest set predictions:")
for i, row in enumerate(test_data[:10]):
    x1, x2, x3, x4, x5, x6, x7, y_actual = row
    y_pred_scaled = predict(x1, x2, x3, x4, x5, x6, x7)

    y_actual_original = y_actual * stds["Population"] + means["Population"]
    y_pred_original = y_pred_scaled * stds["Population"] + means["Population"]

    print(
        f"Row {i+1}: "
        f"Actual = {y_actual_original:.3f}, "
        f"Predicted = {y_pred_original:.3f}"
    )

print(f"\nFinal Train MSE (scaled): {mse(train_data):.4f}")
print(f"Final Test MSE (scaled): {mse(test_data):.4f}")