import numpy as np
from sklearn.cluster import KMeans
import pickle as pik
import networkx as nx
import community

embeddings = np.genfromtxt('facebooknode2vecEmbeddings.csv', delimiter = ",")

kmeans = KMeans(n_clusters=12)

kmeans.fit(embeddings)

with open('facebookEdgeList.txt', 'rb') as f:
    edgeList = pik.load(f)

nG = nx.Graph()
nG.add_nodes_from(range(embeddings.shape[0]))
nG.add_edges_from(edgeList)

partition = dict(enumerate(kmeans.labels_))

nxModularity =  community.modularity(partition,nG)

print("networkx modularity: ", nxModularity)