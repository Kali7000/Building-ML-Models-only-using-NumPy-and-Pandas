
# -*- coding: utf-8 -*-
"""
Ordinal Logistic Regression from Scratch

A pure Python/NumPy implementation of Ordinal Logistic Regression using Stochastic
Gradient Descent, L2 Regularization, and Class Weighting. Designed to predict
wine quality (3-8) based on chemical properties.

Author: Kali
Date: September 21, 2026
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(z):
    """Safely calculates the Sigmoid activation function."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + math.exp(-z))


def standardize(df, saved_stats=None):
    """
    Standardizes a dataframe using Z-score scaling.
    If saved_stats are provided, uses those to scale (crucial for Test Set isolation).
    """
    df_working = df.copy()
    wine_quality = df_working.pop('quality')
    
    if saved_stats is None:    
        saved_stats = {'mean': df_working.mean(), 'std': df_working.std()}
    
    standardized_df = (df_working - saved_stats['mean']) / saved_stats['std']
    standardized_df['quality'] = wine_quality
    
    return standardized_df, saved_stats


def calculate_cross_entropy(predicted_probability_of_target_class):
    """Calculates the loss/penalty for a single prediction."""
    p = max(min(predicted_probability_of_target_class, 1 - 1e-15), 1e-15)
    return -math.log(p)


def build_confusion_matrix(actual_labels, predicted_labels):
    """Prints a formatted 6x6 confusion matrix for Wine Quality (3-8)."""
    print("\n--- Confusion Matrix ---")
    print("Rows: Actual Quality | Columns: Predicted Quality")
    print("      3   4   5   6   7   8")
    print("    -------------------------")
    
    for actual_quality in range(3, 9):
        row_string = f"{actual_quality} | "
        for predicted_quality in range(3, 9):
            count = sum(1 for a, p in zip(actual_labels, predicted_labels) 
                        if a == actual_quality and p == predicted_quality)
            row_string += f"{count:3d} "
        print(row_string)


def train_model(train_wine_df, weights, thresholds, lr, epochs, class_weights, l2_penalty=0.01):  
    """
    Executes Stochastic Gradient Descent to train the ordinal weights and thresholds.
    Includes Learning Rate decay and L2 Regularization.
    """
    train_set_wine_matrix = train_wine_df.to_numpy() 
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Simple Learning Rate Decay
        if epoch == 1000:
            lr = 0.01
        elif epoch == 2500:
            lr = 0.004
        
        for row in train_set_wine_matrix:
            features = row[:-1]
            label_index = int(row[-1]) - 3
            
            # 1. Calculate underlying score
            z_score = np.dot(weights, features)
            
            # 2. Calculate cumulative probabilities against all thresholds
            cumulative_probability = [sigmoid(th - z_score) for th in thresholds]
            cumulative_probability.append(1.0)
            
            # 3. Extract exact probability for the true target class
            if label_index == 0:
                prob_of_correct_ans = cumulative_probability[0]
            else:
                prob_of_correct_ans = cumulative_probability[label_index] - cumulative_probability[label_index - 1]
            
            # 4. Determine Error Gradients
            indicator_above = 1 if label_index < 5 else 0
            indicator_below = 1 if label_index > 0 else 0
            
            prob_above = cumulative_probability[label_index] if label_index < 5 else 1.0
            prob_below = cumulative_probability[label_index - 1] if label_index > 0 else 0.0
            
            # Apply Class Weights to force attention on rare wine scores
            multiplier = class_weights[label_index]
            error_w = ((indicator_above - prob_above) + (indicator_below - prob_below)) * multiplier
            
            # 5. Update Thresholds (Only updating those bordering the true class)
            if label_index < 5:
                error_t_above = (1 - prob_above) * multiplier
                thresholds[label_index] += lr * error_t_above
                
            if label_index > 0:
                error_t_below = (0 - prob_below) * multiplier
                thresholds[label_index - 1] += lr * error_t_below
                
            # Enforce Ordinal Constraint: Thresholds must remain strictly ordered
            thresholds = sorted(thresholds)
    
            # 6. Update Weights
            weights -= (features * lr * error_w)
            
            # Track Loss
            total_loss += calculate_cross_entropy(prob_of_correct_ans)
            
        # Apply L2 penalty (Ridge) once per epoch to prevent overfitting
        weights -= (lr * l2_penalty * weights)
        
        average_loss = total_loss / len(train_set_wine_matrix)
        loss_history.append(average_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Average Log Loss: {average_loss:.4f}")
    
    return weights, thresholds, loss_history


def test_model(test_wine_df, weights, thresholds):
    """Evaluates the trained model against unseen test data."""
    test_wine_matrix = test_wine_df.to_numpy() 
    
    actual_labels = []
    predicted_labels = []
    correct_guesses = 0
    close_guesses = 0
    
    for row in test_wine_matrix:
        features = row[:-1]
        actual_label = int(row[-1])
        label_index = actual_label - 3
        
        z_score = np.dot(weights, features)
        
        cumulative_probability = [sigmoid(th - z_score) for th in thresholds]
        cumulative_probability.append(1.0)
        
        # Calculate the exact probability for all 6 buckets
        bucket_probs = [cumulative_probability[0]] + [
            cumulative_probability[i] - cumulative_probability[i - 1] for i in range(1, 6)
        ]
        
        # Select highest probability bucket
        predicted_index = np.argmax(bucket_probs)
        predicted_label = predicted_index + 3
        
        actual_labels.append(actual_label)
        predicted_labels.append(predicted_label)

        if predicted_label == actual_label:
            correct_guesses += 1
            
        # Track "Off-By-One" accuracy (crucial for Ordinal models)
        if abs(predicted_label - actual_label) <= 1:
            close_guesses += 1

    accuracy = correct_guesses / len(test_wine_df)
    close_accuracy = close_guesses / len(test_wine_df)
    
    print(f"\nStrict Accuracy (Exact Matches): {accuracy * 100:.2f}%")
    print(f"Off-By-One Accuracy:             {close_accuracy * 100:.2f}%")
    
    return actual_labels, predicted_labels


# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    file_path = r"C:\Users\Kali\OneDrive - purdue.edu\Classes\ML\ML Jourey\Logistic regression\Ordinal_logistics_regression\WineQT.csv"
    
    # Load and clean
    wine_df = pd.read_csv(file_path)
    if 'Id' in wine_df.columns:
        wine_df = wine_df.drop(columns=["Id"])
    
    # Shuffle and Split (80/20)
    train_len = int(len(wine_df) * 0.8)
    shuffled_dataset = wine_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_set = shuffled_dataset.iloc[:train_len]   
    test_set = shuffled_dataset.iloc[train_len:]   
    
    # Standardize data safely to prevent test leakage
    train_wine_df, stats = standardize(train_set)
    test_wine_df, _ = standardize(test_set, stats)
    
    # Class weights combat class imbalance by heavily penalizing errors on rare wine scores
    class_weights = {
        0: 10.0,  # Quality 3
        1: 5.0,   # Quality 4
        2: 1.0,   # Quality 5
        3: 1.0,   # Quality 6
        4: 5.0,   # Quality 7
        5: 10.0   # Quality 8
    }
    
    # Initialize Model Parameters
    weights = np.zeros(11)
    thresholds = [-4.0, -2.0, 0.0, 2.0, 4.0]
    lr = 0.05
    epochs = 2200
    lambda_val = 0.01
       
    # Train
    print("--- Commencing Training ---")
    weights, thresholds, loss_history = train_model(
        train_wine_df, weights, thresholds, lr, epochs, class_weights, lambda_val
    )
    
    # Test
    print("\n--- Evaluating Model ---")
    actual_labels, predicted_labels = test_model(test_wine_df, weights, thresholds)
    build_confusion_matrix(actual_labels, predicted_labels)
    
    # Optional: Plot the loss curve
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(range(epochs), loss_history, color='blue')
        plt.title('Ordinal Logistic Regression: Cross-Entropy Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"\nCould not generate plot. Error: {e}")