# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:28:41 2026

@author: Kali
"""

import pandas as pd
import numpy as np
import random



def calulate_weigheted_mse(left, right):
    n_left = len(left)
    n_right = len(right)
    n_total = n_left + n_right
    
    if n_left == 0 or n_right == 0:
        return float('inf')
    
    mse_left = np.var(left)
    mse_right = np.var(right)
    
    return ((n_left/n_total)*mse_left + (n_right/n_total)*mse_right)


def split_finder(df, avilable_features, target_col):
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
    

def build_tree(df, available_features, target, max_depth,feature_fraction=4.0, current_depth = 0):
    #Base conditions to stop the loop    
    if len(df)<15 or current_depth >= max_depth or len(available_features) == 0:
        return np.mean(df[target])
    
    
    # Feature Subsampling: Pick a random subset of features for this specific tree
    num_features = int(len(available_features)*feature_fraction)
    subset_feature = random.sample(available_features,num_features)

    
    best_feature, best_threshold, best_MSE = split_finder(df, subset_feature, target)
    
    #another base case
    if best_feature is None:
        return np.mean(df[target])
        
    left_df = df[df[best_feature]<=best_threshold]
    right_df = df[df[best_feature] > best_threshold]
    
    
    if len(left_df) == 0 or len(right_df) == 0:
        return np.mean(df[target])
    
    #Recursion
    tree = {'feature':best_feature,
            'threshold': best_threshold,
            'left': build_tree(left_df,available_features,target,max_depth,feature_fraction, current_depth+1),
            'right': build_tree(right_df, available_features, target,max_depth,feature_fraction, current_depth+1)}

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



class RandomForestRegressor:
    def __init__(self, n_trees = 50, max_depth = 9, min_samples = 15, feature_fraction=0.4):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_fraction = feature_fraction
        self.forest = []
    
        
    def fit(self, X_train, target_col):
            self.forest = []
            available_features = [col for col in X_train.columns if col != target_col]
            
            print(f"Training a forest of {self.n_trees} trees...")
            
            for i in range(self.n_trees):
                # 1. Bootstrapping: Data sampling
                bootstrapped_df = X_train.sample(frac=1.0, replace=True)
                
                # 2. Train the tree
                tree = build_tree(
                    df=bootstrapped_df, 
                    available_features=available_features, 
                    target=target_col, 
                    max_depth=self.max_depth,
                    feature_fraction=self.feature_fraction
                )
                
                self.forest.append(tree)
                
                if (i+1) % 10 == 0:
                    print(f"Built {i+1}/{self.n_trees} trees...")
    
    
    def predict(self, test_df):
        
        print("Generating predictions...")
        
        prediction_of_every_tree = []
        
        for tree in self.forest:
            tree_predictions = test_df.apply(lambda row: predict_row(tree,row),axis = 1)
            prediction_of_every_tree.append(tree_predictions)
        
        prediction_of_every_tree = np.array(prediction_of_every_tree)  
        print(f"shape of the prediction_of_every_tree {prediction_of_every_tree.shape}")
        
        final_precitions = np.mean(prediction_of_every_tree, axis= 0)
            
        return final_precitions
            
            
            


if __name__ == "__main__":
    daily_df = pd.read_csv("day.csv")
    daily_df.sort_values('instant')
    daily_df.drop(columns = ['instant','dteday','casual', 'registered'],inplace = True)
    daily_df.dropna(inplace=True)
    
    train_len = int(len(daily_df)*0.8)
    train_df = daily_df[:train_len]
    test_df = daily_df[train_len:]
    target_col = 'cnt'

    print("\nBuilding the Tree...")
    # Create the model
    rf = RandomForestRegressor(n_trees=50, max_depth=6, feature_fraction=0.6)
    # Train the forest
    rf.fit(train_df, target_col)
    
    print("\nForest has blossomed!!")
    # Generate the ensemble predictions
    ensemble_preds = rf.predict(test_df)
    # Calculate the new, much lower error!
    errors = np.abs(ensemble_preds - test_df['cnt'].values)
    print(f"\nRandom Forest Average Error: {np.mean(errors):.2f} bikes")
            
            
            
            
        
        
        