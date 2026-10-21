
import pandas as pd
import numpy as np


def normalize(df):
    stats = {
        'mean': df.mean(),
        'std': df.std()
    }

    # 2. Standardize (Z-Score)
    df_z = (df - stats['mean']) / stats['std']
    
    return df_z, stats
 
def normalize_with_stat(df, stat):
    df_z = (df - stat['mean']) / stat['std']  # ← use the parameter
    return df_z
    
def preProcessing(file_name, ordinal_encoding_threshold):

    df = pd.read_csv(file_name)
    column_list = df.columns.tolist()
    str_cols = []
    int_cols = []
    for i in column_list:
        if df[i].dtype == 'object':
            str_cols.append(i)
            #print(f"{i}: {df[i].dtype}")
        else:
            int_cols.append(i)
    
    df = df.dropna(subset = int_cols)
    id_df = df[['Id']]
    df = df.drop(columns = ['Id'])

    
    ##Autometing Encoding 
    
    oneHot_col = []
    
    for obj_col in str_cols:
        no_unique_vals = (df[obj_col].unique()).tolist()
        if len(no_unique_vals) >= ordinal_encoding_threshold:    
            temp_ordinal_dict = {}
            ordinal_num = 1
            for val in no_unique_vals:
                temp_ordinal_dict[val] = ordinal_num
                ordinal_num +=1
                
            df[obj_col] = df[obj_col].map(temp_ordinal_dict)
        else:
            oneHot_col.append(obj_col)
    
    df_processed = pd.get_dummies(df, columns= oneHot_col, drop_first= True, dtype=float)
    
    return df_processed, id_df

def build_features_degree2(column_vals):
    features = []
    for xi in column_vals:
        features.append(xi)   
    for xii in column_vals:
        features.append(xii ** 2)   
    for i in range(len(column_vals)):
        for j in range(i+1, len(column_vals)):
            features.append((column_vals[i])*(column_vals[j]))
    return features


#train
def train_lasso(train_df, lambda_val = 0.5,learning_rate = 0.001, epochs = 1000):

    column_list = train_df.columns.tolist()
    column_list.remove('SalePrice')
    num_col = len(column_list)
     
    #hyperparamaters
    weights = np.zeros((int(((num_col+1)*(num_col+2))/2))-1) #substracting the bias term
    bias = 0.0   
    sale_prices = train_df.pop('SalePrice').values.tolist()
    train_set = train_df.values.tolist()
    
    best_mse = None
    for epoch in range(epochs):
        mse_total = 0.0
        
        for columns, actual_sale_prices in zip(train_set,sale_prices):
            features = build_features_degree2(columns)
            
            predicted_salePrice  = np.dot(features, weights) + bias
            
            error =  predicted_salePrice - actual_sale_prices
        
            features = np.array(features)
            
            #raw gradient #Calculate the raw Nudge (Error + Lasso Penalty)
            raw_gradient = (error * features) + (lambda_val * np.sign(weights))
            
            #clip raw gradient #THE SAFETY NET: Force gradient to stay between -5 and 5
            clipped_gradient = np.clip(raw_gradient, -5.0, 5.0)
            
            # Update weights
            weights = weights - (learning_rate * clipped_gradient)
            
            #Snap-to-Zero for Lasso (Cleans up useless polynomial features)
            weights[np.abs(weights) < 1e-4] = 0.0
            
    
            # Clip bias update as well
            bias -= learning_rate * np.clip(error, -5.0, 5.0)
            
            if epoch% 10 == 0:
                mse_total += (predicted_salePrice - actual_sale_prices) ** 2
             
        
        # Print the loss every 200 epochs to watch it drop!
        if epoch% 10 == 0:
            mse_train = mse_total/len(sale_prices)
            print(f"Epoch {epoch}:  train MSE = {mse_train:.4f}")
            if best_mse == None or best_mse > mse_train:
                best_mse = mse_train
            else:
                pass
                #print(f"Epoch {epoch}: MSE did not improve. Best: {best_mse:.4f}")
                
    
    
    return weights, bias    

    
ordinal_encoding_threshold = 3
train_df, ID_df = preProcessing('train.csv',ordinal_encoding_threshold)

train_df, z_paramaters = normalize(train_df)

salePriceMean = z_paramaters['mean'].pop('SalePrice')
salePriceSTD = z_paramaters['std'].pop('SalePrice')

def deNormalize(val):
    unNormalized_price = (val*salePriceSTD) + salePriceMean
    return unNormalized_price

#hyperparamaters
lambda_val = 0.002
learning_rate = 0.001
epochs = 1000

#training output
weights, bias = train_lasso(train_df, lambda_val,learning_rate, epochs)


#Test
test_df, id_df = preProcessing('test.csv',ordinal_encoding_threshold)

test_df = normalize_with_stat(test_df,z_paramaters)

# ==============================================================
# 3. THE FIX: Align the Test set to match the Train set exactly
# ==============================================================
# This forces test_df to have the exact same columns as train_df.
# If test_df is missing a dummy column, it adds it and fills it with 0.0
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0.0)



test_df = test_df.values.tolist()
salePrice = []
for column_list in test_df:
    features = build_features_degree2(column_list)
    predicted_salePrice  = np.dot(features, weights) + bias
    salePrice.append(deNormalize(predicted_salePrice))

id_df['SalePrice'] = salePrice
