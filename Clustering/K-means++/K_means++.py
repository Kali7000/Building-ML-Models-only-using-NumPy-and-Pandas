# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:36:11 2026
@author: Vishal Wagh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_distance(point1, point2):
    """ Return the Euclidean distance between two point (a point is an row from the dataframe)
    """
    return np.sqrt(np.sum((point1-point2)**2))


def calulate_wcss(input_df, labels, centroids):
    wcss = 0
    
    for i in range(len(centroids)):
        cluster_points = input_df[labels == i]
        
        squared_distances = np.sum((cluster_points - centroids[i])**2)
        
        wcss += squared_distances

    return wcss

def K_means_plus_plus_init(input_features,k):

    ##K-Means++
    centroids = []
    
    first_index = np.random.choice(input_features.shape[0])
    centroids.append(input_features[first_index])
    
    ##loop k-1 times to get new centroids
    for _ in range(1,k):     
        distances = np.array([min([np.sum((point - c) ** 2) for c in centroids]) for point in input_features])
        
        # Convert those distances into probabilities (further points have higher probability)
        probabilities = distances / distances.sum()
        
        # 4. Pick the next centroid using those weighted probabilities
        next_index = np.random.choice(input_features.shape[0], p=probabilities)
        centroids.append(input_features[next_index])
        
    return np.array(centroids)

## the K-maans algorithm
def K_means(input_features, k= 5, max_iters = 100):
    
    #set the seed for reproduciblity
    #np.random.seed(42)
    
    ##K-means++
    centroids = K_means_plus_plus_init(input_features,k)
    
    
    # ##K-means regular
    # #select k distinct random rows from the dataset as our centroids
    # random_indices = np.random.choice(input_features.shape[0],size =k, replace=False)
    # centroids = input_features[random_indices]
    
    #iterate through the epochs
    for iteration in range(max_iters):
        #create a list of lists to store indexes belonging to each cluster
        clusters = [[] for _ in range(k)]
        
        #iterate through the dataset, get the index and the row values at each index
        for indx, point in enumerate(input_features):
            #calculate the distance between each centroid and the current row
            distances = [calculate_distance(point, centroid) for centroid in centroids]
            #select the index {0,1,2,3,4} of the smallest distance
            closest_distance = np.argmin(distances)
            #add the index of the current row to the list of clusters at index closest_distance ex.. if closest_cluster is 2, then the idx will go to [[],[],[idx],...] 
            clusters[closest_distance].append(indx)
         
        #make a copy of the current centroids before updating them to check if it is equal to the previous centroid
        previous_centroids = centroids.copy()
        
        #update the centroids
        
        #iterate through each centroid which is [0,1,2,3,4]
        for i in range(k):
            # Select the list at index i from the list if lists called clusters
            # skip if the cluster is empty
            if len(clusters[i])> 0:
                # Take the mean of all rows and output a an average row which will be the new centroid of that cluster for next intration
                centroids[i] = np.mean(input_features[clusters[i]], axis = 0)
                
        # Check if the updated centroid it is equal to the previous centroid, if it is then we've reached convergence and break the loop
        if np.all(centroids == previous_centroids):
            print(f"Algorith converged at iteration {iteration}")
            break
    # add label to the dataframe
    # crate an numpy array the lenght of the dataset to add it to the df 
    labels = np.zeros(input_features.shape[0])
    #Loop through the clusters, get the index of the clusters {0,1,2,3,4} and the list of indexes in each cluster [3, 45, 2323, 656,...] 
    for cluster_idx, cluster in enumerate(clusters):
        #loop through current cluster and get the value(index)
        for point_idx in cluster:
            # Change the lable at the selceted point index to the cluster index
            labels[point_idx] = cluster_idx
    
    
    return centroids, labels





#Load the CSV file 
dataframe = pd.read_csv("Mall_Customers.csv")
dataframe['Gender_Num'] = dataframe['Gender'].map({'Male': 0, 'Female':1})
#slecet the columns that we want to use
input_df = dataframe[['Gender_Num','Age','Annual Income (k$)','Spending Score (1-100)']].values



wcss_values = []
k_range= range(1,11)

for k in k_range:
    
    centroids, cluster_labels = K_means(input_df, k = k)
    
    current_wcss = calulate_wcss(input_df, cluster_labels, centroids)
    
    wcss_values.append(current_wcss)


# 4. Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss_values, marker='o', linestyle='--', color='b')
plt.title('Elbow Method (Custom Implementation)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(k_range)
plt.grid(True)
plt.show()    



best_k = 5
best_centroids, best_cluster_labels = K_means(input_df, k = best_k)

dataframe['Cluster'] = best_cluster_labels
print(dataframe[['CustomerID', 'Gender','Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head(10))


#create a wide figure to old the graphs
fix, axes = plt.subplots(1,3, figsize=(18,5))

#plot 1: age vs speding score
sns.scatterplot(data=dataframe,x='Age', y='Spending Score (1-100)', hue= 'Cluster', palette= 'Set1', s = 75, ax=axes[0])
axes[0].scatter(centroids[:, 1], centroids[:, 3], marker='*', s=300, c='black', label='Centroid')
axes[0].set_title('Age vs Spending Score')

# Plot 2: Gender vs speding score
sns.stripplot(data=dataframe,x='Gender', y='Spending Score (1-100)', hue= 'Cluster', palette= 'Set1', size =6, ax=axes[1],jitter=True, dodge=True)
axes[1].scatter(centroids[:, 0], centroids[:, 3], marker='*', s=300, c='black', label='Centroid')
axes[1].set_title('Gender vs Spending Score')

#plot 3: Annual income vs spending score
sns.scatterplot(data=dataframe,x='Annual Income (k$)', y='Spending Score (1-100)', hue= 'Cluster', palette= 'Set1', s = 75, ax=axes[2])
axes[2].scatter(centroids[:, 2], centroids[:, 3], marker='*', s=300, c='black', label='Centroid')
axes[2].set_title('Annual Income vs Spending Score')

plt.tight_layout()
plt.show()

