# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:01:30 2026

@author: Kali
"""

import pandas as pd
import math
import numpy as np



email_df = pd.read_csv("emails.csv")
print(email_df.head())
email_df = email_df.drop(columns =["Email No."])

#sort the df columns
email_df = email_df.sort_index(axis = 1)



train_no = int(0.8*email_df.shape[0])

train_set = email_df[0:train_no]
test_set = email_df[train_no:]

print(train_set.shape)
print(test_set.shape)

train_data = train_set.values
test_data = test_set.values


#hyperparamaters
#bias
b = 0
lr =0.005 #learning rate

epochs = 1200

#weights
num_features = train_data.shape[1] - 1
weights = np.zeros(num_features)
    



#activation function
def sigmoid(z):
    # This acts as a speed limit to prevent OverflowErrors
    z = np.clip(z, -500, 500)
    return 1 / (1 + math.exp(-z))

# function to CALCULATE the Log Loss score
def calculate_log_loss(y_actual, y_pred):
    # Adding a tiny number (1e-15) so we never calculate math.log(0), which crashes Python
    y_pred = max(min(y_pred, 1 - 1e-15), 1e-15) 
    if y_actual == 1:
        return -math.log(y_pred)
    else:
        return -math.log(1 - y_pred)


#train

for epoch in range(epochs):
    total_loss =0
    for varaibles, actual in zip(train_data[:,:-1], train_data[:,-1]):
        z = np.dot(weights,varaibles) + b
        y_pred = sigmoid(z)
        
        
        #error
        error = y_pred - actual
        #update weights
        weights= weights - (lr*error*varaibles)
        b = b - (lr * error)
        

        # Monitor: Let's actually calculate the Log Loss to see it!
        total_loss += calculate_log_loss(actual, y_pred)
    # Print the loss every 200 epochs to watch it drop!
    if epoch% 200 == 0:
        print(f"Epoch {epoch} | Total Log Loss: {total_loss:.4f}")


#test
print("Test results")
wrong= 0
total_loss =0
for varaibels, actual in zip (test_data[:,:-1], test_data[:,-1]):
    z = np.dot(weights,varaibels) +b
    prediction = sigmoid(z)
    if prediction<0.5:
        result = 0
    else:
        result = 1
    
    if result != actual:
        wrong+=1
    total_loss += calculate_log_loss(actual, prediction)
    #print(f"Predicted: {result}, actual = {actual}")
print(f"{wrong} got wrong, out of {len(test_data)}")
Accurecy= (wrong/len(test_data))*100
print(f"Accurecy: {Accurecy}%, with Log Loss = {total_loss:.4f} on test data")




list_of_columns = email_df.columns.tolist()
list_of_columns.remove('Prediction')

#predict email

def predict_email(text):
    
    with open(text, 'r') as f:
        my_list = f.read().split()
        content = f.read()
        print(content)
    


    # Create table with one row (index [0]) filled with 0s
    mini_email_df = pd.DataFrame(0, index=[0], columns=list_of_columns)
    
    for word in my_list:
        if word in list_of_columns:
            mini_email_df.at[0, word] +=1
            
    z_score = np.dot(weights,varaibels) +b
    prediction_result = sigmoid(z_score)
    
    if prediction_result<0.5:
        return "Spam"
    else:
        return "Not Spam"


print(predict_email('spam email 1.txt'))

