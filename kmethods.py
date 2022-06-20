import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt 


# Setting random seed for reproducibility, this seed may or may not have been cherry-picked.......

np.random.seed(2019)


# Opening initial data sets
animals = pd.read_csv("animals", sep = " ", header = None)
countries = pd.read_csv("countries", sep = " ", header = None)
fruits = pd.read_csv("fruits", sep = " ", header = None)
veggies = pd.read_csv("veggies", sep = " ", header = None)



# Add the cluster category in a new column
animals['Category'] = 'animals'
countries['Category'] = 'countries'
fruits['Category'] = 'fruits'
veggies['Category'] = 'veggies'

# Join all data together
data = pd.concat([animals, countries, fruits, veggies], ignore_index = True)


# Assigning labels
labels = (pd.factorize(data.Category)[0]+1)


# Dropping labels from dataset
x = data.drop([0, 'Category'], axis = 1).values



# Question 1 - create kmeans algorithm from scratch

def kmeans_clustering(x,k,normalised):
 
        algorithm = "KMeans Clustering"

        # Normalise the data using L2
        if normalised == True:
            x = x/np.linalg.norm(x)
    
        # Randomly initialise the first centroids
        centroids = []
        temp = np.random.randint(x.shape[0], size = k)
        while (len(temp) > len(set(temp))):
            temp = np.random.randint(x.shape[0], size = k)
        for i in temp:
            centroids.append(x[i])



    # Create copies of the centroids for updating
        centroids_old = np.zeros(np.shape(centroids))
        centroids_new = deepcopy(centroids)

    # Create a blank distance and cluster assignment object to hold results
        clusters = np.zeros(x.shape[0])
    # Create an error object
        error = np.linalg.norm(centroids_new - centroids_old)
        num_errors = 0


    # Whilst there is an error value:
        while error != 0:
        
            dist = np.zeros([x.shape[0], k])
        # Add one to the number of errors
            num_errors += 1
            for j in range(len(centroids)):
                dist[:, j] = np.linalg.norm(x - centroids_new[j], axis=1)


        # Calculate the cluster assignment
            clusters = np.argmin(dist, axis = 1)

        # Assign the new copy of centroids to the old centroids object
            centroids_old = deepcopy(centroids_new)

        # Calculate the mean to re-adjust the cluster centroids
            for m in range(k):
                centroids_new[m] = np.mean(x[clusters == m], axis = 0)

        # Re-calculate the error
            error = np.linalg.norm(np.array(centroids_new) - np.array(centroids_old))



    #Assign the final clusters and centroids to new objects
        predicted_clusters = clusters+1
    

        precision = np.zeros(len(x))
        recall = np.zeros(len(x))
        FScore = np.zeros(len(x))



        for i in range(len(x)):
        
            count_Of_Category_In_Cluster = np.count_nonzero(labels[predicted_clusters==predicted_clusters[i]] == labels[i])

            total_In_Category = np.count_nonzero(labels==labels[i])

            cluster_counts = np.count_nonzero(predicted_clusters == predicted_clusters[i])

            #Now to compute
            precision[i] = count_Of_Category_In_Cluster / cluster_counts

            recall[i] = count_Of_Category_In_Cluster / total_In_Category

            FScore[i] = 2*(precision[i]* recall[i])/(precision[i] + recall[i])



        precision = (np.sum(precision) / len(x))
        recall = (np.sum(recall) / len(x))
        FScore = (np.sum(FScore) / len(x))
   
    
        print(f'K-Means algorithm results with normalisation set to {normalised}')
        print(f'For K = {k} B-CUBED precision is {(precision)}')
        print(f'For K = {k} B-CUBED recall is {(recall)}')
        print(f'For K = {k} B-CUBED F-score is {(FScore)}')
        print('\n')
    
        return precision,recall,FScore,algorithm,normalised

        
  

    

    
    
# Question 2 - Implement kmedians clustering algorithm from scratch

def kmedians_clustering(x, k, normalised):

    algorithm = "KMedians Clustering"
 
    # Normalise the data using L2
    if normalised == True:
        x = x / np.linalg.norm(x)
    
    # Randomly initialise the first centroids
    centroids = []
    temp = np.random.randint(x.shape[0], size = k)
    while (len(temp) > len(set(temp))):
        temp = np.random.randint(x.shape[0], size = k)
    for i in temp:
        centroids.append(x[i])



    # Create copies of the centroids for updating
    centroids_old = np.zeros(np.shape(centroids))
    centroids_new = deepcopy(centroids)

    # Create a blank distance and cluster assignment object to hold results
    clusters = np.zeros(x.shape[0])
    # Create an error object
    error = np.linalg.norm(centroids_new - centroids_old)
    num_errors = 0


    # Whilst there is an error value:
    while error != 0:
        
        dist = np.zeros([x.shape[0], k])
        # Add one to the number of errors
        num_errors += 1
        for j in range(len(centroids)):
            dist[:, j] = np.sum(np.abs(x - centroids_new[j]), axis=1)


        # Calculate the cluster assignment
        clusters = np.argmin(dist, axis = 1)

        # Assign the new copy of centroids to the old centroids object
        centroids_old = deepcopy(centroids_new)

        # Calculate the mean to re-adjust the cluster centroids
        for m in range(k):
            centroids_new[m] = np.median(x[clusters == m], axis = 0)

        # Re-calculate the error
        error = np.linalg.norm(np.array(centroids_new) - np.array(centroids_old))



    #Assign the final clusters and centroids to new objects
    predicted_clusters = clusters+1
    

    precision = np.zeros(len(x))
    recall = np.zeros(len(x))
    FScore = np.zeros(len(x))



    for i in range(len(x)):
        
        count_Of_Category_In_Cluster = np.count_nonzero(labels[predicted_clusters==predicted_clusters[i]] == labels[i])

        total_In_Category = np.count_nonzero(labels==labels[i])

        cluster_counts = np.count_nonzero(predicted_clusters == predicted_clusters[i])

        #Now to compute
        precision[i] = count_Of_Category_In_Cluster / cluster_counts

        recall[i] = count_Of_Category_In_Cluster / total_In_Category

        FScore[i] = 2*(precision[i]* recall[i])/(precision[i] + recall[i])



    precision = (np.sum(precision) / len(x))
    recall = (np.sum(recall) / len(x))
    FScore = (np.sum(FScore) / len(x))
   
    
    print(f'K-Medians algorithm results with normalisation set to {normalised}')
    print(f'For K = {k} B-CUBED precision is {np.round(precision,2)}')
    print(f'For K = {k} B-CUBED recall is {np.round(recall,2)}')
    print(f'For K = {k} B-CUBED F-score is {np.round(FScore,2)}')
    print('\n')
  
    return precision,recall,FScore,algorithm,normalised




 
def scorePlot(precision,recall,fscore,kCount,algorithm,normalised):

    #Giving plot a more appropriate style
    plt.style.use('ggplot')

    #Plotting scores against K values
    plt.plot(kCount, precision, label="Precision")
    plt.plot(k_Count, recall, label="Recall")
    plt.plot(k_Count, fscore, label="F-Score")
    
    #Setting rest of the chart up
    if normalised == True:
        plt.title(f'{algorithm} with L2 normalisation applied')
    else:
        plt.title(f'{algorithm}')
   

    plt.xlabel('K value')
    plt.ylabel("Score")

    plt.legend()
    plt.show() 







# Questions 3-6
## For question 3, set to kmeans_clustering(x,k,False) and set to kmeans_clustering(x,k,True) for question 4
### For question 5 set to kmedians_clustering(x,k,False) and set to kmedians_clustering(x,k,True) for question 6


k_Count = []
p_Count = []
r_Count = []
f_Count = []

'''for k in range(1,10):
    
    k_Count.append(k)

    Precision,Recall,FScore,algorithm,normalised = kmeans_clustering(x,k,True)
    p_Count.append(Precision)
    r_Count.append(Recall)
    f_Count.append(FScore)

    scorePlot(p_Count,r_Count,f_Count,k_Count,algorithm,normalised)'''


kmeans_clustering(x,4,False)