# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:48:40 2026

@author: Kali
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load the data directly from a public URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Let's peek at the first 5 rows to see what the data looks like
print("--- First 5 rows of our dataset ---")
print(dataset.head())
print("\n")


# X contains the 4 measurements (the first 4 columns)
X = dataset.values[:, 0:4]

# y contains the species name (the last column)
y = dataset.values[:, 4]

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# Create the model. 
# multi_class='multinomial' tells it to use the Softmax approach we learned!
# solver='lbfgs' is just the specific math optimization tool it uses to do gradient descent.
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Train the model on our 80% training data
model.fit(X_train, y_train)
print("Model training complete!\n")



# Make predictions on the hidden test data
predictions = model.predict(X_test)

# Calculate how many it got right
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Show a detailed breakdown of how it did on each of the 3 species
print("--- Detailed Report ---")
print(classification_report(y_test, predictions))