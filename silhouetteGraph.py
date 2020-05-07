from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


#X,y = make_blobs(n_samples = 4000, centers = 12, n_features = 5)
#X = np.genfromtxt('facebooknode2vecEmbeddings.csv', delimiter = ",")
X = np.genfromtxt('facebookPoincareEmbeddings.csv', delimiter=",")
distances = np.genfromtxt('facebookPoincareDistanceMatrix.csv', delimiter=',')


distortions = []
for i in range(2, 60):
    clustering = AgglomerativeClustering(n_clusters=i, affinity="precomputed", linkage="complete").fit(distances)
    silScoreHyp = silhouette_score(distances, clustering.labels_, metric='precomputed')
    silScore = silhouette_score(X, clustering.labels_)
    print("Iteration " , i-1 , "complete.")
    distortions.append(silScore)

print("Max: ", max(distortions))
print("# of clusters: ", distortions.index(max(distortions)) + 2)




fig = plt.figure(figsize=(15, 5))
plt.plot(range(2,60), distortions, 'bo-')
plt.grid(True)
plt.title('idk')
plt.show()