# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 00:44:43 2026

@author: Kali
"""

import pandas as pd
import math 
import numpy as np

def standardize(df):
    df_scaled = df.copy()
    
    # Select only numeric columns
    num_cols = df_scaled.select_dtypes(include=['number']).columns
    
    # Apply Z-score formula: (x - mean) / std
    df_scaled[num_cols] = (df_scaled[num_cols] - df_scaled[num_cols].mean()) / df_scaled[num_cols].std()
    
    return df_scaled




def softmax(z_scores):
    # Calculate e^z for every score
    exponentials = [math.exp(z) for z in z_scores]
    total_sum = sum(exponentials)
    
    # Divide each by the total to get probabilities that add up to 1.0
    
    probabilities = [e/total_sum for e in exponentials]
    return probabilities


def calculate_cross_entropy(predicted_probablity_of_target_class):
    # We only penalize based on the probability assigned to the ACTUAL correct answer
    
    p = max(min(predicted_probablity_of_target_class, 1 - 1e-15),1e-15)
    
    return -math.log(p)




# Load the data directly from a public URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Let's peek at the first 5 rows to see what the data looks like
print("--- First 5 rows of our dataset ---")
print(dataset.head())
print("\n")
print(dataset['class'].unique())


w0 = [0.0,0.0,0.0]
w1 = [0.0,0.0,0.0]
w2 = [0.0,0.0,0.0]
w3 = [0.0,0.0,0.0]
b = [0.0,0.0,0.0]
lr = 0.015
epochs = 5000


train_len = int(len(dataset)*0.8)

shuffled_dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
train_set = shuffled_dataset.iloc[:train_len]   # Takes from index 0 up to train_len (80%)
test_set = shuffled_dataset.iloc[train_len:]   # Takes from train_len to the very end (20%)


standerdized_df = standardize(train_set)


for epoch in range(epochs):
    total_loss = 0
    
    for row in standerdized_df.itertuples(index=False):
        # Extract features by order/name
        x0, x1, x2, x3, y = row[0], row[1], row[2], row[3], row[4]
        
        z_scores = [
        w0[0]*x0 + w1[0]*x1 + w2[0]*x2 + w3[0]*x3 + b[0],
        w0[1]*x0 + w1[1]*x1 + w2[1]*x2 + w3[1]*x3 + b[1],
        w0[2]*x0 + w1[2]*x1 + w2[2]*x2 + w3[2]*x3 + b[2] ]
    
        y_predicted_probabilities = softmax(z_scores)
        # print(y_predicted_probabilities)
        
        if y == 'Iris-setosa':
            index = 0 
        
        elif y == 'Iris-versicolor':
            index = 1
        else:
            index = 2
        
        probablity_of_correct_ans = y_predicted_probabilities[index]
        
        total_loss += calculate_cross_entropy(probablity_of_correct_ans)
        
        # Step B & C: Update the weights for ALL 3 categories
        
        for catogery_index in range(3):
            # Is this category the actual correct answer? (1 if yes, 0 if no)
            
            target = 1 if catogery_index == index else 0
            
            
            error = y_predicted_probabilities[catogery_index] - target
            
            #update weights and biases
            w0[catogery_index] -= (lr*error*x0)
            w1[catogery_index] -= (lr*error*x1)
            w2[catogery_index] -= (lr*error*x2)
            w3[catogery_index] -= (lr*error*x3)
            b[catogery_index] -= (lr*error)
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Total Cross-Entropy Loss: {total_loss:.4f}")

print("\n--- Training Complete ---\n")

catagory_class = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

##Test the model

    
standerdized_test_set = standardize(test_set)

correct_guesses = 0 

for row in standerdized_test_set.itertuples(index=False):
    # Extract features by order/name
    x0, x1, x2, x3, y = row[0], row[1], row[2], row[3], row[4]
    
    z_scores = [
    w0[0]*x0 + w1[0]*x1 + w2[0]*x2 + w3[0]*x3 + b[0],
    w0[1]*x0 + w1[1]*x1 + w2[1]*x2 + w3[1]*x3 + b[1],
    w0[2]*x0 + w1[2]*x1 + w2[2]*x2 + w3[2]*x3 + b[2] ]

    y_predicted_probabilities = softmax(z_scores)
    
    winning_index = y_predicted_probabilities.index(max(y_predicted_probabilities))
    predicted_name = catagory_class[winning_index]
    
    
    if y == predicted_name:
        correct_guesses+=1
    else:
        print(f"Actual = {y},   Predicted: {predicted_name}")
    
    
print(f"\nAccuracy: {correct_guesses / len(standerdized_test_set) * 100:.2f}%")
    
        
        
        
        
        
