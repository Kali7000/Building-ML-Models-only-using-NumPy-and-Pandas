# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 00:14:25 2026
@author: Vishal Wagh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#laod the data
features = pd.read_csv("data.csv")
#remove null columns
features = features.dropna(how = 'all', axis = 1)
#pop target variable and incode
target = features.pop('diagnosis')
target = target.map({'M':0,'B':1})
#remove noice columun
features = features.drop(columns = 'id')

#get mean and standerd deveation of every column
data_means = np.mean(features, axis=0)
data_stds = np.std(features, axis=0)
#standeridise the dataset
standardized_data = (features-data_means)/data_stds
#get the covariene matrix
co_varience_matrix = np.cov(standardized_data,rowvar=False)

# give the co-varience matrix to eigh function which takes the perfectly 
# symmitrical co-varince matrix and returns the eigenvalues and eigenvectors 
# of each of the features
eigenvalues, eigenvactors = np.linalg.eigh(co_varience_matrix)

#sort the eigenvalues array and get the sorted indicies
sorted_indexes = np.argsort(eigenvalues)[::-1]
#use the sorted indicies to sort the eigenvectors and eigenvalues
eigenvalues = eigenvalues[sorted_indexes]
eigenvactors = eigenvactors[:,sorted_indexes]

# take the top 7 eigenvectors that represent the most amount of data
eigenvactors_subset = eigenvactors[:,0:7]

#project the data onto the eigenvectors and remove rest
PCAed_data = np.dot(standardized_data,eigenvactors_subset)


# calulate the varience of each PCA
percentage_eigenvalues = eigenvalues/np.sum(eigenvalues)
cumulative_variance = np.cumsum(percentage_eigenvalues)

#plot
plt.figure(figsize = (10,6))
plt.bar(range(1,len(percentage_eigenvalues)+1),percentage_eigenvalues,
        alpha = 0.7, align='center', label ='PCA impact of each feature')
plt.step(range(1,len(cumulative_variance)+1), cumulative_variance,
         where='mid', color='red', label = 'Cumulative varience')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.title('Scree Plot: Information Retained by PCA')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

