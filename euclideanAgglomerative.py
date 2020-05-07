import networkx as nx
import community
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pickle as pik


embeddings = np.genfromtxt('facebooknode2vecEmbeddings.csv', delimiter = ",")

#default is ward linkage, can change to complete (to match hyperbolic) if we want
clustering = AgglomerativeClustering(n_clusters=12).fit(embeddings)

with open('facebookEdgeList.txt', 'rb') as f:
    edgeList = pik.load(f)

nG = nx.Graph()
nG.add_nodes_from(range(embeddings.shape[0]))
nG.add_edges_from(edgeList)

partition = dict(enumerate(clustering.labels_))

nxModularity =  community.modularity(partition,nG)

print("networkx modularity: ", nxModularity)