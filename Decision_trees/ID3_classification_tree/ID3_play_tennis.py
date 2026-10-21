# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 02:07:45 2026

@author: Kali
"""

import math
import pandas as pd

def calculate_entropy(target_column):
    """
    Calculates the Entropy of a list of target values (e.g., ['Yes', 'No', 'Yes', ...])
    """
    total_items = len(target_column)
    if total_items == 0:
        return 0.0
    
    # 1. Count how many times each class appears (e.g., {'Yes': 9, 'No': 5})
    class_counts = {}
    for value in target_column:
        if value not in class_counts:
            class_counts[value] = 0
        class_counts[value] += 1
        
    # 2. Apply the mathematical formula: -Sum(p * log2(p))
    entropy = 0.0
    for count in class_counts.values():
        probability = count / total_items
        # Log2 of 0 is mathematically undefined, so we skip it if probability is 0
        if probability > 0:
            entropy -= probability * math.log2(probability)
            
    return entropy

def calculate_information_gain(df, feature_name, target_name):
    """
    Calculates how much Information Gain we get by splitting the dataset on `feature_name`.
    """
    # 1. Calculate H(D) - The original entropy of the parent dataset
    total_entropy = calculate_entropy(df[target_name].tolist())
    
    # 2. Find the unique categories in the feature (e.g., ['Sunny', 'Overcast', 'Rain'])
    unique_values = df[feature_name].unique()
    
    # 3. Calculate the weighted entropy of the split subsets
    total_items = len(df)
    subset_entropy = 0.0
    
    for value in unique_values:
        # Create the smaller bucket (subset) where the feature equals this specific value
        subset = df[df[feature_name] == value]
        
        # Calculate the entropy of this new, smaller bucket
        subset_target_column = subset[target_name].tolist()
        h_subset = calculate_entropy(subset_target_column)
        
        # Weight it by its size (|Dv| / |D|) and add it to our total subset entropy
        weight = len(subset) / total_items
        subset_entropy += weight * h_subset
        
    # 4. Information Gain = Original Entropy - Subset Entropy
    information_gain = total_entropy - subset_entropy
    
    return information_gain

def build_tree(df, target_name, available_features):
    """
    Recursively builds an ID3 Decision Tree using pandas DataFrames.
    """
    target_column = df[target_name].tolist()
    
    # -------------------------------------------------------------
    # BASE CASE 1: Is the bucket completely pure?
    # -------------------------------------------------------------
    # If every single row in this dataset has the exact same answer (e.g., all 'Yes')
    # we don't need to ask any more questions. Just return the answer!
    unique_answers = list(set(target_column))
    if len(unique_answers) == 1:
        return unique_answers[0]  # Returns 'Yes' or 'No'
        
    # -------------------------------------------------------------
    # BASE CASE 2: Are we out of questions to ask?
    # -------------------------------------------------------------
    # If we have used up every feature (Outlook, Wind, Humidity, Temp)
    # but the bucket is STILL mixed, we just guess the most common answer.
    if len(available_features) == 0:
        # Find the most frequent item in the list
        most_common_answer = max(set(target_column), key=target_column.count)
        return most_common_answer
        
    # -------------------------------------------------------------
    # THE GREEDY CHOICE: Which feature gives us the highest score?
    # -------------------------------------------------------------
    best_feature = None
    highest_gain = -1.0
    
    # YOUR TURN: Loop through `available_features`. 
    # Call calculate_information_gain() for each one.
    # Save the feature that produces the highest score into `best_feature`.
    
    for fature in available_features:
        info_gain = calculate_information_gain(df, fature, target_name)
        if info_gain > highest_gain:
            highest_gain = info_gain
            best_feature = fature
    
    
    # -------------------------------------------------------------
    # THE RECURSION: Build the branch and drill down!
    # -------------------------------------------------------------
    # Start the dictionary for this branch
    tree = {best_feature: {}}
    
    # We used this feature, so remove it from the list of available questions
    remaining_features = [f for f in available_features if f != best_feature]
    
    # Find the unique categories for the best feature (e.g., Sunny, Overcast, Rain)
    unique_categories = df[best_feature].unique()
    
    for category in unique_categories:
        # Filter the DataFrame to ONLY rows that match this category
        subset_df = df[df[best_feature] == category]
        
        # MAGIC: We call THIS EXACT FUNCTION again, but we feed it
        # the smaller subset_df and the remaining_features list!
        branch_result = build_tree(subset_df, target_name, remaining_features)
        
        # Attach whatever that function returned (either a 'Yes' or a whole nested dictionary)
        # to our current tree.
        tree[best_feature][category] = branch_result
        
    return tree



play_tennis_df = pd.read_csv(r"C:\Users\Kali\OneDrive - purdue.edu\Classes\ML\ML Jourey\Decision_trees\ID3\PlayTennis.csv")


# Create a list of the features ['Outlook', 'Temperature', 'Humidity', 'Wind']
feature_columns = [col for col in play_tennis_df.columns if col != 'play']

# Call the function!
my_decision_tree = build_tree(play_tennis_df, target_name='play', available_features=feature_columns)

import pprint
pprint.pprint(my_decision_tree)


