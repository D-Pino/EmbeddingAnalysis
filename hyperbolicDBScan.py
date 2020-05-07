import hdbscan
import numpy as np
import networkx as nx
import community
import pickle as pik


embeddings = np.genfromtxt('twitchRU64PoincareEmbeddings.csv', delimiter = ",")

clusterer = hdbscan.HDBSCAN(metric = 'precomputed')

distances = np.genfromtxt('twitchRU64PoincareDistanceMatrix.csv', delimiter=',')

clusterer.fit(distances)

print('DBScan rundown: ')
print('Labels: ', clusterer.labels_)
print('Number of clusters: ', clusterer.labels_.max() + 1)
print('Probabilities: ',clusterer.probabilities_)

count = 0
for x in clusterer.labels_:
    if(x==-1):
        count+=1
print("Number of noise points: ", count)

#alter labels???
c = clusterer.labels_.max() + 1;
for i in range(clusterer.labels_.size):
    if(clusterer.labels_[i]==-1):
        clusterer.labels_[i]=c;
        c+=1;
print("New max: ", clusterer.labels_.max())





with open('twitchRUEdgeList.txt', 'rb') as f:
    edgeList = pik.load(f)

nG = nx.Graph()
nG.add_nodes_from(range(embeddings.shape[0]))
nG.add_edges_from(edgeList)

partition = dict(enumerate(clusterer.labels_))

nxModularity = community.modularity(partition,nG)

print("networkx modularity: ", nxModularity)