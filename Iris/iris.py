from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
 
data = load_iris()
df = data.data
df = df[:,1:3]
 
agg_results = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
resultlab = agg_results.fit_predict(df)
 
plt.figure(figsize = (8,5))
plt.scatter(df[resultlab == 0 , 0] , df[resultlab == 0 , 1] , c = 'red')
plt.scatter(df[resultlab == 1 , 0] , df[resultlab == 1 , 1] , c = 'blue')
plt.scatter(df[resultlab == 2 , 0] , df[resultlab == 2 , 1] , c = 'green')
plt.show()

from scipy.cluster.hierarchy import dendrogram , linkage

Z = linkage(df, method = 'ward')

dendro = dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()