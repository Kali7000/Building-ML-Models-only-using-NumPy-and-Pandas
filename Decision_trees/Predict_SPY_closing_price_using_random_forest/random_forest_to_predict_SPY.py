# -*- coding: utf-8 -*-
"""
@author: Kali
"""
import yfinance as yf
import pandas as pd
import numpy as np

def best_split(x, y, feature_idx, min_leaf):
    best = (None,None,np.inf)  # (feature, threshold, error)
    n = len(y)
    for col in feature_idx:
        #X[:, j] is feature column j. argsort returns the row positions that would sort 
        #it from smallest to largest.
        order = np.argsort(x[:,col])
        
        #Once sorted, every possible split is just "first k rows go left, 
        #the rest go right," which lets us score all thresholds in one pass.
        sorted_input, sorted_target = x[order,col], y[order]
        
        #Running totals:
        # csum[i] = sum of ys[0..i]
        # csum2[i] = sum of the squares of ys[0..i]
        #ex. For sorted_target = [1, 2, 3]: csum = [1, 3, 6], csum2 = [1, 5, 14].
        csum, csum2 = np.cumsum(sorted_target), np.cumsum(sorted_target**2)
        
        total, total2 = csum[-1], csum2[-1]
        
        i = np.arange(min_leaf - 1, n - min_leaf)
        
        #Row counts on each side, for every candidate split: left has i + 1 rows, 
        #right has the remainder.
        n_l, n_r = i + 1, n - (i + 1)
        
        
        # Left part: csum2[i] - csum[i]**2 / n_l, using the running sums up to row i.
        # Right part: (total2 - csum2[i]) - (total - csum[i])**2 / n_r
        ## SSE (sum of squared errors)
        sse = (csum2[i] - csum[i]**2 / n_l) + ((total2 - csum2[i]) - (total - csum[i])**2 / n_r)
        
        sse = np.where(sorted_input[i] != sorted_input[i + 1], sse, np.inf)
        
        if len(sse) == 0 or np.isinf(sse).all():
            continue
        
        #the best split for this feature.
        k = np.argmin(sse)

        if sse[k] < best[2]:
            best = (col, (sorted_input[i[k]] + sorted_input[i[k] + 1]) / 2, sse[k])
        
    return best


def build_tree(input_df, target, max_depth, min_leaf, num_features, random_generator, current_depth = 0):
    # Stop condition
    if current_depth>= max_depth or len(target) < 2*min_leaf:
        return target.mean()
    
    subset_features = random_generator.choice(input_df.shape[1], size = num_features, replace = False)
    
    #split the tree
    best_feature, best_threshold, best_MSE = best_split(input_df, target, subset_features, min_leaf)
    
    if best_feature is None:
        return target.mean()
    
    #left right split
    # True for rows whose feature "best_feature" is at or below the threshold (they go left), 
    # False otherwise.
    mask = input_df[:,best_feature] <= best_threshold

    #X[mask] and y[mask] are the left rows, ~mask (the opposite) are the right rows
    ##Recursion
    return {'feature':best_feature,
            'threshold': best_threshold,
            'left':build_tree(input_df[mask], target[mask], max_depth, min_leaf, num_features, random_generator, current_depth+1),
            'right':build_tree(input_df[~mask], target[~mask], max_depth, min_leaf, num_features, random_generator, current_depth+1)}


def predict_row(tree, row):
    #while loop insted of recursion
    while isinstance(tree, dict):
        tree = tree['left'] if row[tree['feature']] <= tree['threshold'] else tree['right']
    return tree   



def mse(y, p): return np.mean((y - p)**2)
def dir_acc(y, p): return np.mean(np.sign(y) == np.sign(p))



class RandomForestRegressor:
    def __init__(self, n_trees = 100, max_depth = 4, min_samples = 50, feature_fraction=0.5, seed = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_fraction = feature_fraction
        self.seed = seed
            
    def fit(self, X_train, target_col):
        random_generator = np.random.default_rng(self.seed)
        
        #How many features each split is allowed to look at.
        num_features = max(1, int(X_train.shape[1]*self.feature_fraction))
        self.forest = []

        print(f"Training a forest of {self.n_trees} trees...")
        
        for i in range(self.n_trees):
            # 1. Bootstrapping: Data sampling
            #Generate random integers/indexes
            bootstrapped_indexes = random_generator.integers(0,len(target_col),len(target_col)) # rng.integers(low, high, size)
            
            # 2. Train the tree
            tree = build_tree(
                X_train[bootstrapped_indexes],  
                target_col[bootstrapped_indexes], 
                self.max_depth,
                self.min_samples,
                num_features,
                random_generator
            )
            
            self.forest.append(tree)
            
            if (i+1) % 10 == 0:
                print(f"Built {i+1}/{self.n_trees} trees...")
    

    def predict(self, X):
        print("Generating predictions...")
        preds = np.array([[predict_row(t, r) for r in X] for t in self.forest])
        return preds.mean(axis=0)



if __name__ == "__main__":
    ## Data prepration
    stock_data = yf.download("SPY", start="2010-01-01", auto_adjust=True)
    
    # print(stock_data.head())
    # print(stock_data.tail())
    # print(stock_data.info())
    stock_data.columns = stock_data.columns.get_level_values(0)
    stock_data = stock_data.sort_index()
    
    close, vol = stock_data['Close'], stock_data['Volume']
    
    ##building the features
    
    #create an empty datafrme the size of stock_data with the same index column
    features = pd.DataFrame(index = stock_data.index)
    
    #this crates a new columns "ret_1" that saves the percent return from yesterday to today
    # pct_change(1) computes (today - 1 day ago) / 1 day ago. 
    # This is today's return versus yesterday. For example, closes of 100 then 101 give 0.01 (+1%). 
    # The first row is NaN because it has no previous day.
    features['ret_1'] = close.pct_change(1)
    # 2 days ago
    features['ret_2'] = close.pct_change(2)
    #5 days ago and so on
    features['ret_5']  = close.pct_change(5)
    features['ret_10'] = close.pct_change(10)
    
    ##this adds a new column 'vol_10' which holds the standerd devation of price changes
    # of "close" column over last 10 days
    features['vol_10'] = close.pct_change().rolling(10).std()
    
    # Today's price divided by its 20-day moving average. 
    # Above 1 means the price is above its recent average (trending up), 
    # below 1 means below it. Again it's a ratio, so it doesn't depend on the price level.
    features['ma_ratio'] = close / close.rolling(20).mean()
    
    # shift(-1) moves the series up by one row, so each row now holds tomorrow's close. 
    # Dividing by today's close and subtracting 1 gives tomorrow's return.
    # y1: return from today to the next trading day
    target = pd.DataFrame({'y1':close.shift(-1)/close - 1}) 
    
    data = pd.concat([features, target], axis=1).dropna()

    
    #Train test split
    
    split = int(len(data)*0.8)
    train = data.iloc[:split-1]         # drop last 2 rows: their labels reach into the test period
    test = data.iloc[split:]
    
    cols = list(features.columns)
    X_train, X_test = train[cols].values, test[cols].values
    Y_train, Y_test = train['y1'].values, test['y1'].values
    
    
    # Create the model
    randomForest1 = RandomForestRegressor(n_trees = 80, max_depth = 4, min_samples = 20, feature_fraction=0.33, seed = 42)
    # Train the forest
    randomForest1.fit(X_train, Y_train)
    print("\nForest has blossomed!!")

    p = randomForest1.predict(X_test)
    print("Forest   MSE:", mse(Y_test, p),                    " dir acc:", dir_acc(Y_test, p))
    print("Mean     MSE:", mse(Y_test, Y_train.mean()),      " dir acc:", dir_acc(Y_test, np.ones(len(Y_test))))

    
    
    

