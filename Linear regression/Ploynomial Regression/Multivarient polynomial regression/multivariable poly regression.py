# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:34:29 2026

@author: Kali
"""

import pandas as pd

import matplotlib.pyplot as plt

algea_df = pd.read_csv('algeas.csv', usecols= ['Light','Nitrate','Iron','Phosphate','Temperature','pH','CO2','Population'])

print(algea_df.head())



# NEW: Save means and standard deviations so we can use them later for testing!
means = algea_df[['Light','Nitrate','Iron','Phosphate','Temperature','pH','CO2','Population']].mean()
stds = algea_df[['Light','Nitrate','Iron','Phosphate','Temperature','pH','CO2','Population']].std()


# Feature Scaling (Standardization)
for col in ['Light','Nitrate','Iron','Phosphate','Temperature','pH','CO2','Population']:
    algea_df[col] = (algea_df[col] - means[col]) / stds[col]

print(algea_df.head())


algea_df = algea_df.sample(frac = 1)
print(algea_df.shape)
train_rows = int(0.8*algea_df.shape[0])


train_set= algea_df[0:train_rows]
test_set = algea_df[train_rows:]


print(train_set.shape)
print(test_set.shape)

train_data = train_set.values.tolist()
test_data = test_set.values.tolist()


## numner of features = n - variables, d= degree of the polynomial, B = Binomial
# numner =B((n+d)/d) - 1 

# Initialize all weights to 0.0
w0 = 0.0

# Linear (7)
w1 = w2 = w3 = w4 = w5 = w6 = w7 = 0.0

# Squared (7)
w8 = w9 = w10 = w11 = w12 = w13 = w14 = 0.0

# Interaction (21)
w15 = w16 = w17 = w18 = w19 = w20 = 0.0
w21 = w22 = w23 = w24 = w25 = 0.0
w26 = w27 = w28 = w29 = 0.0
w30 = w31 = w32 = 0.0
w33 = w34 = 0.0
w35 = 0.0
learning_rate = 0.02
epochs = 2000

def predict(x1, x2, x3, x4, x5, x6, x7):
    return (
        # Bias
        w0

        # Linear terms
        + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7

        # Squared terms
        + w8*x1**2 + w9*x2**2 + w10*x3**2 + w11*x4**2
        + w12*x5**2 + w13*x6**2 + w14*x7**2

        # Interaction terms
        + w15*x1*x2 + w16*x1*x3 + w17*x1*x4 + w18*x1*x5 + w19*x1*x6 + w20*x1*x7
        + w21*x2*x3 + w22*x2*x4 + w23*x2*x5 + w24*x2*x6 + w25*x2*x7
        + w26*x3*x4 + w27*x3*x5 + w28*x3*x6 + w29*x3*x7
        + w30*x4*x5 + w31*x4*x6 + w32*x4*x7
        + w33*x5*x6 + w34*x5*x7
        + w35*x6*x7
    )

# ----------------------------
# Evaluation helpers
# ----------------------------
def mse(data):
    total = 0.0
    for x1, x2, x3, x4, x5, x6, x7, y_actual in data:
        y_pred = predict(x1, x2, x3, x4, x5, x6, x7)
        total += (y_pred - y_actual) ** 2
    return total / len(data)

def predict_original_scale(x1, x2, x3, x4, x5, x6, x7):
    """
    Returns the prediction in the ORIGINAL Population scale,
    not the standardized scale.
    """
    y_pred_scaled = predict(x1, x2, x3, x4, x5, x6, x7)
    y_pred_original = y_pred_scaled * stds["Population"] + means["Population"]
    return y_pred_original

# ----------------------------
# Training loop with loss printout
# ----------------------------
for epoch in range(epochs):
    for x1, x2, x3, x4, x5, x6, x7, y_actual in train_data:
        y_pred = predict(x1, x2, x3, x4, x5, x6, x7)
        error = y_pred - y_actual

        # Bias
        w0 -= learning_rate * error

        # Linear
        w1 -= learning_rate * error * x1
        w2 -= learning_rate * error * x2
        w3 -= learning_rate * error * x3
        w4 -= learning_rate * error * x4
        w5 -= learning_rate * error * x5
        w6 -= learning_rate * error * x6
        w7 -= learning_rate * error * x7

        # Squared
        w8  -= learning_rate * error * (x1 ** 2)
        w9  -= learning_rate * error * (x2 ** 2)
        w10 -= learning_rate * error * (x3 ** 2)
        w11 -= learning_rate * error * (x4 ** 2)
        w12 -= learning_rate * error * (x5 ** 2)
        w13 -= learning_rate * error * (x6 ** 2)
        w14 -= learning_rate * error * (x7 ** 2)

        # Interactions
        w15 -= learning_rate * error * (x1 * x2)
        w16 -= learning_rate * error * (x1 * x3)
        w17 -= learning_rate * error * (x1 * x4)
        w18 -= learning_rate * error * (x1 * x5)
        w19 -= learning_rate * error * (x1 * x6)
        w20 -= learning_rate * error * (x1 * x7)

        w21 -= learning_rate * error * (x2 * x3)
        w22 -= learning_rate * error * (x2 * x4)
        w23 -= learning_rate * error * (x2 * x5)
        w24 -= learning_rate * error * (x2 * x6)
        w25 -= learning_rate * error * (x2 * x7)

        w26 -= learning_rate * error * (x3 * x4)
        w27 -= learning_rate * error * (x3 * x5)
        w28 -= learning_rate * error * (x3 * x6)
        w29 -= learning_rate * error * (x3 * x7)

        w30 -= learning_rate * error * (x4 * x5)
        w31 -= learning_rate * error * (x4 * x6)
        w32 -= learning_rate * error * (x4 * x7)

        w33 -= learning_rate * error * (x5 * x6)
        w34 -= learning_rate * error * (x5 * x7)

        w35 -= learning_rate * error * (x6 * x7)

    # Print training progress every 100 epochs
    if epoch % 100 == 0:
        train_loss = mse(train_data)
        test_loss = mse(test_data)
        print(f"Epoch {epoch}: train MSE = {train_loss:.4f}, test MSE = {test_loss:.4f}")

# ----------------------------
# Final test results
# ----------------------------
print("\nFinal model parameters:")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)
print("w3 =", w3)
print("w4 =", w4)
print("w5 =", w5)
print("w6 =", w6)
print("w7 =", w7)
print("w8 =", w8)
print("w9 =", w9)
print("w10 =", w10)
print("w11 =", w11)
print("w12 =", w12)
print("w13 =", w13)
print("w14 =", w14)
print("w15 =", w15)
print("w16 =", w16)
print("w17 =", w17)
print("w18 =", w18)
print("w19 =", w19)
print("w20 =", w20)
print("w21 =", w21)
print("w22 =", w22)
print("w23 =", w23)
print("w24 =", w24)
print("w25 =", w25)
print("w26 =", w26)
print("w27 =", w27)
print("w28 =", w28)
print("w29 =", w29)
print("w30 =", w30)
print("w31 =", w31)
print("w32 =", w32)
print("w33 =", w33)
print("w34 =", w34)
print("w35 =", w35)

print("\nTest set predictions:")
for i, row in enumerate(test_data[:10]):
    x1, x2, x3, x4, x5, x6, x7, y_actual = row
    y_pred_scaled = predict(x1, x2, x3, x4, x5, x6, x7)

    # Convert back to original Population scale
    y_actual_original = y_actual * stds["Population"] + means["Population"]
    y_pred_original = y_pred_scaled * stds["Population"] + means["Population"]

    print(
        f"Row {i+1}: "
        f"Actual = {y_actual_original:.3f}, "
        f"Predicted = {y_pred_original:.3f}"
    )

print(f"\nFinal Train MSE (scaled): {mse(train_data):.4f}")
print(f"Final Test MSE (scaled): {mse(test_data):.4f}")