# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:44:46 2026

@author: Kali
"""

import pandas as pd
import numpy as np
import math


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + math.exp(-z))


def standardize(df, saved_stats = None):
    
    df_working = df.copy()
    wine_quality = df_working.pop('quality')
    
    if saved_stats == None:    
        saved_stats = {'mean': df_working.mean(), 'std': df_working.std()}
        #print(saved_stats)
    
    standerdized_df = (df_working - saved_stats['mean'])/saved_stats['std']
    
    standerdized_df['quality'] = wine_quality
    
    return standerdized_df, saved_stats



def calculate_cross_entropy(predicted_probablity_of_target_class):
    # We only penalize based on the probability assigned to the ACTUAL correct answer
    
    p = max(min(predicted_probablity_of_target_class, 1 - 1e-15),1e-15)
    
    return -math.log(p)


def build_confusion_matrix(actual_labels, predicted_labels):
    print("\n--- Confusion Matrix ---")
    print("Rows: Actual Quality | Columns: Predicted Quality")
    print("      3   4   5   6   7   8")
    print("    -------------------------")
    
    # We have 6 categories (3 through 8)
    for actual_quality in range(3, 9):
        row_string = f"{actual_quality} | "
        
        for predicted_quality in range(3, 9):
            # Count how many times this specific Actual/Predicted combo happened
            count = 0
            for a, p in zip(actual_labels, predicted_labels):
                if a == actual_quality and p == predicted_quality:
                    count += 1
                    
            # Format the number so it aligns nicely in the grid
            row_string += f"{count:3d} "
            
        print(row_string)



def train_model(train_wine_df, weights, thresholds, lr, epochs, l2_penalty = 0.001):  
    train_set_wine_matrix = train_wine_df.to_numpy() 
    for epoch in range(epochs):
        total_loss = 0
        
        if epoch == 1000:
            lr = 0.01
        elif epoch == 2500:
            lr = 0.004
        
        for row in train_set_wine_matrix:
            features = row[:-1]
            lable_index = int(row[-1])-3
            
            z_score = np.dot(weights,features)
            
            comulative_probablity = []
            
            for th in thresholds:
                probablity = sigmoid(th - z_score)
                comulative_probablity.append(probablity)
            
            comulative_probablity.append(1.0)
            
            if lable_index == 0:
                prob_of_correct_ans = comulative_probablity[0]
            else:
                prob_of_correct_ans = comulative_probablity[lable_index]- comulative_probablity[lable_index - 1]
            
            
            indicator_above = 1 if lable_index < 5 else 0
            indicator_below = 1 if lable_index > 0 else 0
            
            prob_above = comulative_probablity[lable_index] if lable_index < 5 else 1.0
            prob_below = comulative_probablity[lable_index - 1] if lable_index > 0 else 0.0
            
    
            error_w = (indicator_above - prob_above) + (indicator_below - prob_below)
            # ... [your threshold updates] ...
            if lable_index < 5:
                    error_t_above = 1 - prob_above
                    thresholds[lable_index] += lr * error_t_above
                
            if lable_index > 0:
                error_t_below = 0 - prob_below
                thresholds[lable_index - 1] += lr * error_t_below
                
            # NEW: Force the thresholds to stay strictly ordered
            thresholds = sorted(thresholds)
    
    
            #update weights
            weights -= (features*lr*error_w)
            
            total_loss += calculate_cross_entropy(prob_of_correct_ans)
        #Apply the L2 penalty once at the end of the epoch
        weights -= (lr * l2_penalty * weights)
        if epoch%100 == 0:
            print(f"For Epoch: {epoch}|| Total log loss:{total_loss:.4f}")
    
    return weights, thresholds
            
            
        
        
        
     


def test_model(test_wine_df, weights, thresholds):
    test_wine_matrix = test_wine_df.to_numpy() 
    
    actual_labels = []
    predicted_labels = []
    correct_guesses = 0
    close_guesses = 0
    
    for row in test_wine_matrix:
        features = row[:-1]
        actual_lable = int(row[-1])
        lable_index = actual_lable-3
        
        z_score = np.dot(weights,features)
        
        comulative_probablity = []
        
        for th in thresholds:
            probablity = sigmoid(th - z_score)
            comulative_probablity.append(probablity)
        
        comulative_probablity.append(1.0)
        
        if lable_index == 0:
            prob_of_correct_ans = comulative_probablity[0]
        else:
            prob_of_correct_ans = comulative_probablity[lable_index]- comulative_probablity[lable_index - 1]
                
            # 3. Calculate the exact probability for all 6 buckets
        bucket_probs = []
        bucket_probs.append(comulative_probablity[0]) # Bucket 0
        for i in range(1, 6):
            bucket_probs.append(comulative_probablity[i] - comulative_probablity[i - 1])
        
        # 4. Find the bucket with the highest probability
        # np.argmax returns the index of the highest number in a list
        predicted_index = np.argmax(bucket_probs)
        
        predicted_label = predicted_index + 3
        
        actual_labels.append(actual_lable)
        predicted_labels.append(predicted_label)
            

        # 1. Count exact matches (Strict Accuracy)
        if predicted_label == actual_lable:
            correct_guesses += 1
            
        # 2. Count matches that are off by 1 or less
        # abs() gets the absolute value, so 5-6 and 6-5 both equal 1
        distance = abs(predicted_label - actual_lable)
        if distance <= 1:
            close_guesses += 1

    # Calculate Strict accuracy
    accuracy = correct_guesses / len(test_wine_df)
    print(f"Strict Accuracy: {accuracy * 100:.2f}%")
    
    # Calculate Off-by-One accuracy
    close_accuracy = close_guesses / len(test_wine_df)
    print(f"Off-by-One Accuracy: {close_accuracy * 100:.2f}%")
    
    return actual_labels, predicted_labels, correct_guesses


file_path = r"C:\Users\Kali\OneDrive - purdue.edu\Classes\ML\ML Jourey\Logistic regression\Ordinal_logistics_regression\WineQT.csv"

wine_df = pd.read_csv(file_path)
stats = {}

wine_df = wine_df.drop(columns =["Id"])

# for col in wine_df.columns.tolist():
#     print(f"{col} : {wine_df[col].isna().sum()}")
    
#     print(f"{wine_df[col].describe()}")


train_len = int(len(wine_df)*0.8)
shuffled_dataset = wine_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_set = shuffled_dataset.iloc[:train_len]   # Takes from index 0 up to train_len (80%)
test_set = shuffled_dataset.iloc[train_len:]   # Takes from train_len to the very end (20%)

train_wine_df, stats = standardize(train_set)

# We pass the training stats in to ensure the test set is scaled identically
test_wine_df, stats = standardize(test_set, stats)


weights = np.zeros(11)
thresholds = [-4.0, -2.0, 0.0, 2.0, 4.0]
lr = 0.00001
epochs = 1000
lambda_val = 0.0001
   
weights, thresholds = train_model(train_wine_df, weights, thresholds, lr, epochs,lambda_val)

actual_labels, predicted_labels, correct_guesses = test_model(test_wine_df, weights, thresholds)

build_confusion_matrix(actual_labels, predicted_labels)
