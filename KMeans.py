# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:28:07 2016

@author: Trace
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets

def kmeans(data, k, threshold, max_iter):
    """Inputs: data - array on which to perform clustering 
    target - actual clusters to which data belong
    k - number of clusters (default=3)
    threshold - difference between new centroids and old centroids (stopping criterion) (default=0.001) 
    max_iter - maximum number of iterations to run (stopping criterion) (default=19)
    Outputs: centroids - array of k centroids
    closest - array indicating to which cluster/index an observation belongs
    accuracy - percentage of observations in correct groups"""
    
    def initialize_centroids(data, k):
        """returns k centroids from the initial points"""
        chunk_size = round(data.shape[0]/k)
        copy = data.copy()
        np.random.shuffle(copy)
        chunks = np.array([copy[x:x+chunk_size] for x in range(0, copy.shape[0], chunk_size)])
        return np.array([chunks[x].mean(axis=0) for x in range(chunks.shape[0])])
    
    def closest_centroid(data, centroids):
        """returns an array containing the index to the nearest centroid for each point"""
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def new_centroids(data, closest, centroids):
        """returns the new centroids assigned from the points closest to them"""
        return np.array([data[closest==x].mean(axis=0) for x in range(centroids.shape[0])])
    
    # Initialize the number of iterations and threshold stopping criteria variables
    n_iter = 0
    epsilon = threshold + 1
    
    # Initialize the centroids using the function defined above.
    centroids = initialize_centroids(data.values, k)
    
    # Iterate through until either the max number of iterations is met 
    # or the threshold is no longer exceeded.
    while (n_iter <= max_iter) and (epsilon>threshold):
        # Increase the iterations counter by 1.
        n_iter = n_iter + 1
        # Sort the centroids        
        centroids = np.sort(centroids, axis=0)
        # Create array indicating to which cluster each observation belongs.
        closest = closest_centroid(data.values, centroids)
        # Create array of new centroids based on previously determined clusters.
        new = new_centroids(data.values, closest, centroids)
        # Calculate the new threshold measurement.
        epsilon = np.array([np.linalg.norm(new[x] - centroids[x]) for x in range(new.shape[0])]).sum()
        # Set old centroids equal to new centroids.
        centroids = new
    
    # Add the cluster labels to the dataframe
    data['cluster label'] = closest
    
    # Create a summary dataframe that will hold the number of iterations,
    # epsilon, and size of each cluster
    summary = pd.DataFrame({'iterations': n_iter, 'epsilon': epsilon}, index=np.arange(0,1))
    
    for x in range(k):
        label = 'group ' + str(x)
        summary[label] = np.unique(closest, return_counts=True)[1][x]
        
    sns_plot = sns.pairplot(data, hue = 'cluster label')
    #sns_plot.savefig('pairplot.png')
    
    #data.to_csv('data with cluster label.csv', index=False)
    # Return the data and summary dataframes    
    return data, summary, sns_plot
    
def accuracy(closest, target):
    """Inputs: 
    target - actual clusters to which data belong
    closest - array indicating to which cluster/index an observation belongs
    Outputs:
    accuracy - percentage of observations in correct groups"""
    return ((closest-target)==0).sum() / closest.shape[0]

# Load the two ellipses dataset from the text file
te_data = pd.read_csv('twoEllipsesData.txt', sep=' ', header=-1)
# The dataset doesn't have column names or a description, so they get a generic name.
te_data.columns = ['column 1', 'column 2']
 
# Initialize the inputs to the function
# We have 2 ellipses, and the stopping criteria were defined in the problem.
te_k = 2
te_max_iter = 19
te_threshold = 0.001
 
# Call the kmeans function
te_data, te_summary, te_plot = kmeans(te_data, te_k, te_threshold, te_max_iter)

te_plot.savefig('Two Ellipses Pairplot.png')
te_data.to_csv('Two Ellipses with Cluster Label.csv', index=False)
    
print(te_summary)
    
# Call the iris dataset
iris = datasets.load_iris()
# Extract the observations from the dataset
iris_data = pd.DataFrame(iris.data)
iris_feature_names = iris.feature_names
iris_data.columns = iris_feature_names
# Extract the target clusters from the dataset
iris_target = iris.target

# Initialize the inputs to the function
# We have 3 types of iris, and the stopping criteria were defined in the problem.
iris_k = 3
iris_max_iter = 19
iris_threshold = 0.001

# Call the kmeans function
iris_data, iris_summary, iris_plot = kmeans(iris_data, iris_k, iris_threshold, iris_max_iter)
iris_accuracy = accuracy(iris_data['cluster label'].values, iris_target)

iris_plot.savefig('Iris Pairplot.png')
iris_data.to_csv('Iris with Cluster Label.csv', index=False)

print(iris_summary, '\n', 'Accuracy: ' + str(iris_accuracy))