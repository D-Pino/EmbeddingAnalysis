import networkx as nx
import community
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = pd.read_csv('facebooknode2vecEmbeddings.csv',sep = ',', header = None)
X = X.sort_values(by = 0)
X = X.reset_index()
X = X.drop(["index",0], axis = 1)
embeddings = X.to_numpy()

kmeans = KMeans(n_clusters=12)
kmeans.fit(embeddings)

partitionK = dict(enumerate(kmeans.labels_))



df = pd.read_csv('facebook.csv')
df = df.drop('weight', axis = 1)

edgeList = []

for index, row in df.iterrows():
    edgeList.append((row[0],row[1]))


nG = nx.Graph()
nG.add_nodes_from(range(embeddings.shape[0]))
nG.add_edges_from(edgeList)
nxModularityK = community.modularity(partitionK,nG)

print("KMeans Modularity: ", nxModularityK)

clustering = AgglomerativeClustering(n_clusters=12).fit(embeddings)

partitionA = dict(enumerate(clustering.labels_))

nxModularityA = community.modularity(partitionA,nG)

print("Agglomerative Modularity: ", nxModularityA)



