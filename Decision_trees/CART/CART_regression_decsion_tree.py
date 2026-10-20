# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:03:28 2026
@author: Kali
"""

import pandas as pd
import numpy as np



def calulate_weigheted_mse(left, right):
    n_left = len(left)
    n_right = len(right)
    n_total = n_left + n_right
    
    if n_left == 0 or n_right == 0:
        return float('inf')
    
    mse_left = np.var(left)
    mse_right = np.var(right)
    
    return ((n_left/n_total)*mse_left + (n_right/n_total)*mse_right)



def split_finder(df, avilable_features, target_col = 'cnt'):

    best_feature = None
    best_threshold = None
    best_MSE = float('inf')
    
    for col in avilable_features:
        unique_vals = df[col].unique()
        
        for val in unique_vals:
            subset_MSE = calulate_weigheted_mse((df.loc[df[col] <= val, target_col]), (df.loc[df[col] > val, target_col]))
            
            if subset_MSE < best_MSE:
                best_feature = col
                best_MSE = subset_MSE
                best_threshold = val
    
    return best_feature, best_threshold, best_MSE
    

def build_tree(df, avilable_features, target, max_depth = 5, current_depth = 0):

    #Base conditions to stop the loop    
    if len(df)<15 or current_depth >= max_depth or len(avilable_features) == 0:
        return np.mean(df[target])

               
    best_feature, best_threshold, best_MSE = split_finder(df, avilable_features, target)
    
    #another base case
    if best_feature is None:
        return np.mean(df[target])
    
    
    left_df = df[df[best_feature]<=best_threshold]
    right_df = df[df[best_feature] > best_threshold]
    
    
    if len(left_df) == 0 or len(right_df) == 0:
        return np.mean(df[target])
    
    #recursion
    tree = {'feature':best_feature,
            'threshold': best_threshold,
            'left': build_tree(left_df,avilable_features,target,max_depth,current_depth+1),
            'right': build_tree(right_df, avilable_features, target,max_depth,current_depth+1)}
    
    return tree
    
 

def predict_row(tree,row):
    
    #Base case: if the current node is not in dict/ it is a leaf node (a number)
    if not isinstance(tree, dict):
        return tree


    feature = tree['feature']
    threshold = tree['threshold']
    
    if row[feature]<= threshold:
        return predict_row(tree['left'], row)
    else:
        return predict_row(tree['right'], row)
    
    

def predict(tree, test_df, target):
    predictions = test_df.apply(lambda row: predict_row(tree,row),axis = 1)
    
    
    # Calculate overall Error (Mean Absolute Error for display)
    errors = np.abs(np.array(predictions) - np.array(test_df[target]))
    print(f"\nAverage Prediction Error: {np.mean(errors):.2f} bikes")
    
    return predictions.values
    


def print_tree(tree, indent=""):
    # Base Case: If it's a leaf node (a float), print the final prediction
    if not isinstance(tree, dict):
        print(f"{indent}--> Predict: {tree:.2f} bikes")
        return
    
    feature = tree['feature']
    threshold = tree['threshold']
    
    # Print the Left branch (True condition)
    print(f"{indent}If {feature} <= {threshold:.4f}:")
    print_tree(tree['left'], indent + "   |")
    
    # Print the Right branch (False condition)
    print(f"{indent}Else ({feature} > {threshold:.4f}):")
    print_tree(tree['right'], indent + "   |")
    

if __name__ == "__main__":

    daily_df = pd.read_csv("day.csv")
    daily_df.sort_values('instant')
    daily_df.drop(columns = ['instant','dteday','casual', 'registered'],inplace = True)
    daily_df.dropna(inplace=True)
    #print(daily_df.head())
    
    # ### this did not help at all
    # #One-Hot Encode Categorical Variables
    # categorical_cols = ['season', 'weathersit', 'weekday', 'mnth']
    # daily_df = pd.get_dummies(daily_df, columns=categorical_cols, dtype=int)
    
    # print("New columns after One-Hot Encoding:")
    # print(daily_df.columns.tolist())
    
    
    
    
    train_len = int(len(daily_df)*0.8)
    
    train_df = daily_df[:train_len]
    test_df = daily_df[train_len:]
    
    
    target = 'cnt'
    max_depth = 7
    
    avilable_features = train_df.columns.tolist()
    avilable_features.remove(target)

    print("\nBuilding the Tree...")
    tree = build_tree(train_df, avilable_features, target,max_depth)
    
    print("\nTree is complete!!")
    
    print_tree(tree)
    
    print("\n\nTesting...")
    predict(tree, test_df, target)
    
    

