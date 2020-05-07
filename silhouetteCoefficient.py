from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering


X,y = make_blobs(n_samples = 4000, centers = 12, n_features = 5)

kmeans1 = KMeans(n_clusters=12)
kmeans1.fit(X)

silScoreBlobs = silhouette_score(X,kmeans1.labels_)

print('Blobs silhouette: ', silScoreBlobs)

eucEmbeddings = np.genfromtxt('facebooknode2vecEmbeddings.csv', delimiter = ",")

kmeans2 = KMeans(n_clusters=12)
kmeans2.fit(eucEmbeddings)

silScoreEuc = silhouette_score(eucEmbeddings,kmeans2.labels_)

print('Euc silhouette: ', silScoreEuc)

hypEmbeddings = np.genfromtxt('facebookPoincareEmbeddings.csv', delimiter=",")

distances = np.genfromtxt('facebookPoincareDistanceMatrix.csv', delimiter=',')

clustering = AgglomerativeClustering(n_clusters=12, affinity="precomputed", linkage="complete").fit(distances)

silScoreHyp = silhouette_score(distances, clustering.labels_, metric = 'precomputed')

print('Hyp silhouette: ', silScoreHyp)