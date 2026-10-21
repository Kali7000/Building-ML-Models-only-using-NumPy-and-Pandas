# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:32:34 2026

@author: Kali
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Data
raw_data = pd.read_csv("insurance.csv", usecols=['age','sex','bmi','children','smoker','charges'])

sex_num= []
for sex in raw_data['sex']:
    if sex == 'male':
        sex_num.append(1)
    else:
        sex_num.append(0)

raw_data['sex'] = sex_num   

raw_data['smoker'] = [1 if s == 'yes' else 0 for s in raw_data['smoker']]


      
# NEW: Save means and standard deviations so we can use them later for testing!
means = raw_data[['age', 'bmi', 'children']].mean()
stds = raw_data[['age', 'bmi', 'children']].std()


# Feature Scaling (Standardization)
for col in ['age', 'bmi', 'children']:
    raw_data[col] = (raw_data[col] - means[col]) / stds[col]

print(raw_data.head())


raw_data = raw_data.sample(frac = 1)
print(raw_data.shape[0])
train_roes = int(0.8*raw_data.shape[0])


train_set= raw_data[0:train_roes]
test_set = raw_data[train_roes:]


print(train_set.shape[0])
print(test_set.shape[0])

train_data = train_set.values.tolist()
test_data = test_set.values.tolist()


# 2. Hyperparameters
w_age, w_bmi, w_children, w_sex, w_smoker = 1.0, 1.0, 1.0, 1.0, 1.0
b = 1.0 
LR = 0.001 
epochs = 1000
n = len(train_data) # Number of data points for calculating MSE

mse_history = [] # To track the learning curve


# 3. Learning Loop (Stochastic Gradient Descent)
for epoch in range(epochs):
    epoch_squared_error = 0 

    for x1, x4, x2, x3, x5, y_actual in train_data:
        # Predict
        y_predicted = w_age*x1 + w_bmi*x2 + w_children*x3 + w_sex*x4 + w_smoker*x5 + b
        
        # Calculate error
        error = y_predicted - y_actual
        epoch_squared_error += (error ** 2) 
        
        # Update weights and bias
        w_age -= (LR * error * x1)
        w_bmi -= (LR * error * x2)
        w_children -= (LR * error * x3)
        w_sex -= (LR * error * x4)
        w_smoker -= (LR * error * x5)
        b -= (LR * error)
        
    # Calculate Mean Squared Error for the whole epoch
    mse = epoch_squared_error / n
    mse_history.append(mse) # Store it for plotting
        
    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"epoch: {epoch:4d} | MSE = {mse:.2f}")

print(f"\nFinal parameters -> w_age = {w_age:.2f}, w_bmi = {w_bmi:.2f}, w_child = {w_children:.2f} ; b = {b:.2f}")



# --- 4. Test the Model (Fixed Scaling) ---
for age,sex, bmi, children,  smoker, actual_cost in test_data:
    
    
    # We MUST scale the test data using the exact same mean/std from the training set
    #age_scaled = (age - means['age']) / stds['age']
    #bmi_scaled = (bmi - means['bmi']) / stds['bmi']
    #children_scaled = (children - means['children']) / stds['children']
    
    predicted_cost = (w_age*age) + (w_bmi*bmi) + (w_children*children) + (w_sex*sex) + (w_smoker*smoker) + b
    print(f"Predicted cost for {age}yo, BMI {bmi}, {children} kids, Male{sex}, Smoker{smoker}: ${predicted_cost:,.2f}")
    print(f"Error: {predicted_cost - actual_cost}")


# --- 5. Visualizations ---
# Generate final predictions for the scatter plot
actual_charges = [row[5] for row in train_data]
predicted_charges = [(w_age*row[0] + w_bmi*row[2] + w_children*row[3] + w_sex*row[1] + w_smoker*row[4] + b) for row in train_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: The Learning Curve (Loss over time)
ax1.plot(range(epochs), mse_history, color='purple', linewidth=2)
ax1.set_title('Learning Curve (MSE over Epochs)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mean Squared Error')
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Actual vs. Predicted Performance
ax2.scatter(actual_charges, predicted_charges, alpha=0.5, color='teal')
# Draw the "Perfect Prediction" diagonal line
ax2.plot([min(actual_charges), max(actual_charges)], [min(actual_charges), max(actual_charges)], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
ax2.set_title('Model Performance: Actual vs. Predicted')
ax2.set_xlabel('Actual Insurance Charges ($)')
ax2.set_ylabel('Predicted Charges ($)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()