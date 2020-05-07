import hdbscan
import numpy as np
import networkx as nx
import community
import pickle as pik


embeddings = np.genfromtxt('facebooknode2vecEmbeddings.csv', delimiter = ",")

clusterer = hdbscan.HDBSCAN()
clusterer.fit(embeddings)

print('DBScan rundown: ')
print('Labels: ', clusterer.labels_)
print('Number of clusters: ', clusterer.labels_.max())
print('Probabilities: ',clusterer.probabilities_)

count = 0
for x in clusterer.labels_:
    if(x==-1):
        count+=1
print("Number of noise points: ", count)

with open('facebookEdgeList.txt', 'rb') as f:
    edgeList = pik.load(f)

nG = nx.Graph()
nG.add_nodes_from(range(embeddings.shape[0]))
nG.add_edges_from(edgeList)

partition = dict(enumerate(clusterer.labels_))

nxModularity = community.modularity(partition,nG)

print("networkx modularity: ", nxModularity)