import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score

def largest_cluster(X,labels ,centers):
    lrgs = np.array([])
    for ci,c in enumerate(centers):
        dis= euclidean_distances(X[labels==ci], [centers[ci]])
        sse =  np.sum(dis)
        #print(sse)
        lrgs =np.append(lrgs,sse)
        
    return np.argmax(lrgs)
    
def bkmeans(X,iter_,k):
    data = X.copy()
    cls_labels = np.zeros(data.shape[0])
    cls_centers = np.empty([0,X.shape[1]])
    cls_centers.reshape(0,data.shape[1])
    
    #cluster index to split
    cls = 0
    
    for i in range(k-1):
        
        km = KMeans(n_clusters=2,random_state=42,n_init = 1, max_iter=iter_).fit(data[cls_labels==cls])
        c_preds = km.predict(data[cls_labels==cls])
        
        data_index = data[cls_labels==cls].index
        if len(cls_centers>0):
            cls_labels[cls_labels > cls] += 1
        cls_labels[data_index] = c_preds + cls
        
        if len(cls_centers>0):
            cls_centers = np.delete(cls_centers,cls,axis=0)
        cls_centers = np.insert(cls_centers, cls, km.cluster_centers_,axis=0)
        
        # select largest cluster for spliting
        cls = largest_cluster(data,cls_labels, cls_centers)
        
        
    return cls_labels.astype("int64")
#.reshape(-1,1)          
