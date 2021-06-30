import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, mpld3
import pandas as pd
import os
from scipy.spatial import distance

datapath = os.getcwd()
datapath = datapath + "/data/Seeds_Data/"  # MMI data folder
if not os.path.exists(datapath):
    print("Data Path does not exist")


# In[9]:


def k_mean_algo(data, k=3, tolerance=0.001, max_iterations=100):
    SSE_Sum = 0
    # Ten Iteration for calculating avg SSE
    for outerLoop in range(10):

        numberInstance, numberFeature = data.shape
        # print ("Number of Instance: ", numberInstance, "Number of Feature: ",numberFeature)

        # Take the random Centroid initialization
        indexes = np.random.choice(numberInstance - 1, k, replace=False)
        centroids = {}
        j = 0
        for i in indexes:
            centroids[j] = data[i]
            j = j + 1

        # --------------------------------//
        # print (centroids)

        # begin iterations
        SSE_Old = 0;
        for i in range(max_iterations):
            k_clusters = {}
            for i in range(k):
                k_clusters[i] = []

            # find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                normalized_distances = [distance.euclidean(features, centroids[centroid]) for centroid in centroids]
                classification = normalized_distances.index(min(normalized_distances))
                k_clusters[classification].append(features)

            # average the cluster datapoints to re-calculate the centroids
            SSE_new = 0;
            for classification in k_clusters:
                #                 print ("Cluster: ",k_clusters[classification])
                #                 print ("Cluster size: ",len(k_clusters[classification]))
                #                 print ("Centroid: ",centroids[classification])
                centroids[classification] = np.average(k_clusters[classification], axis=0)
                # Calculate the SSE in One clsuter
                normalized_distances = [distance.euclidean(points, centroids[classification]) for points in
                                        k_clusters[classification]]
                # sum the sse value for one cluster
                for error in normalized_distances:
                    SSE_new = SSE_new + error
            # print ("SSE: ", SSE_new ,"\n\n")

            if (abs(SSE_Old - SSE_new) < tolerance):
                #  print ("SSE terminate")
                break;

            SSE_Old = SSE_new;

        # print("Iteration: ",outerLoop, "SSE:",SSE_new)
        SSE_Sum = SSE_Sum + SSE_new
    # outer loop for 10 iterations
    # sum SSE new there and avg
    print("K Mean Clustering with K=", k, "and average SSE=", SSE_Sum / 10.0)


# In[10]:


seeds_dataset = datapath + 'seeds.txt'
data = pd.read_csv(seeds_dataset, sep="   ", header=None, engine='python')
data.columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']

X = data.values  # returns a numpy array

k_mean_algo(X, 3)
k_mean_algo(X, 5)
k_mean_algo(X, 7)
