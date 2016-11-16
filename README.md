# CPSC605-Assignment-4
Implementation of K-Means Clustering Algorithm in Python

Computer Science- CpSc 405/605

DATA MINING AND DATA ANALYSIS
Assignment#5: (10 points) 
Due by: Monday, Nov. 14. 8:00 AM

Objectives: 
2. Use models to implement data mining tasks.
 
Instructions: 
	In this assignment you will implement the KMeans algorithm, you can use R or Python to write your source code. Name your source code file as KMeans.R or KMeans.py. Run your source code on the Iris dataset, you can find the dataset in the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Iris), read the dataset description to discover how many clusters (K) you should ask for when you run your implementation.

For initialization, use any random function to set the initial clusters’ centers, k points.
For stopping criteria use the following two criteria: 
1)	The squared Euclidean distances between the old mean and the current mean € is less than a threshold, say 0.001.
2)	The number of iterations is less than 20.

Submission: You should submit two files:
1)	Your KMeans source code file, name it as: KMeans.R or KMeans.py
2)	A report file which includes: the size of each cluster, the number of iterations, the final €, the cluster label for each data point, and the accuracy of your algorithm.

___________________________________________________________________________________________________________________
Update 11/13/2016

Computer Science- CpSc 405/605

DATA MINING AND DATA ANALYSIS
Assignment#5: (10 points) 
Due by: Monday, Nov. 14. 8:00 AM

Objectives: 
2. Use models to implement data mining tasks.
 
Instructions: 
	In this assignment you will implement the KMeans algorithm, you can use R or Python to write your source code. Name your source code file as KMeans.R or KMeans.py. Run your source code on the Iris dataset, you can find the dataset in the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Iris), read the dataset description to discover how many clusters (K) you should ask for when you run your implementation.

For initialization, use any random function to set the initial clusters’ centers, k points.
For stopping criteria use the following two criteria: 
1)	The squared Euclidean distances between the old mean and the current mean € is less than a threshold, say 0.001.
2)	The number of iterations is less than 20.

Submission: You should submit two files:
1)	Your KMeans source code file, name it as: KMeans.R or KMeans.py
2)	A report file which includes: the size of each cluster, the number of iterations, the final €, and the cluster label for each data point.





 
SourceCodeSkeleton.R
##################################################
dataFile = "E:\\Assignment5\\twoEllipsesData.txt" #set the dataset path…attached the twoEllipsesData.txt datafile
K=2 #number of clusters
maxIter=20 #max iterations
epsilon=0.00001
X=as.matrix(read.table(dataFile)) # put data in a matrix

N=dim(X)[1]	# number of rows		
d=dim(X)[2]	# number of columns		

C=X[1:K,]#initial K clusters (centers) the first k points

myCluster = matrix(0,1,N)#This is to hold the cluster assignment for each point..
cc = c("red","black")#to plot the points of each cluster in a different color


for(iter in 1:maxIter) # Iterate until a stopping condition is met.
{     
  #for every point, find its distances to all the cluster centers.
 for (i in 1:N)
  {
   #find closest cluster to point i... 
    
 
 
  }
  #for every cluster, calculate the new mean,,
 

  #Calculate the delta, the sum of the differences between the old means and the new means..
  Delta= 
  
  
  
  #Check if delta is less than epsilon, if so open a report file and write the following results:
 
  if (Delta < epsilon | iter==maxIter)
  {
     plot(1,1,xlim=c(min(X[,1]),max(X[,1])), ylim=c(min(X[,2]),max(X[,2])), type="n") # plot the clusters
     for(cl in 1:K) # for each cluster
     {
       #Append to the report file the cluster label of each point.  
       #Append to the report file the size (how many points).
       #Append to the report file the final epsilon and number of iteration
       
       
       
     }
     
     iter=maxIter+1 # break the loop
   }
}
