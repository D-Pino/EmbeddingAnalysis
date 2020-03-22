from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import igraph as ig
import torch as t
import networkx as nx
import community

#load model into dictionary, access the embeddings, convert to numpy array
#facebookLORENTZ.pth
embeddings = t.load('facebook.pth')['embeddings'].numpy()
lorentzEmbeddings = t.load('facebookLORENTZ.pth')['embeddings'].numpy()


distances = np.genfromtxt('lorentzDistanceMatrix.csv', delimiter=',')

#get edgelist from somewhere
df = pd.read_csv('facebook.csv')
df = df.drop('weight', axis = 1)

edgeList = []

for index, row in df.iterrows():
    edgeList.append((row[0],row[1]))

nG = nx.Graph()
nG.add_nodes_from(range(embeddings.shape[0]))
nG.add_edges_from(edgeList)

g = ig.Graph()
g.add_vertices(embeddings.shape[0])
g.add_edges(edgeList)

clustering = AgglomerativeClustering(n_clusters=12, affinity="precomputed", linkage="complete").fit(distances)

igModularity = g.modularity(clustering.labels_,None)

partition = dict(enumerate(clustering.labels_))

nxModularity =  community.modularity(partition,nG)

#print("iGraph modularity: ", igModularity)
print("networkx modularity: ", nxModularity)


'''modularities = []
for i in range(2,50):
    print("Iteration : ", i+1)
    clustering = AgglomerativeClustering(n_clusters = i,affinity = "precomputed", linkage = "complete").fit(distances)
    modularities.append(g.modularity(clustering.labels_,None))


print(modularities)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2,50), modularities, 'bo-')
plt.grid(True)
plt.title('Some kinda curve')
plt.show()'''


#print(clustering.labels_)
'''print("Labels: ")
for x in range(clustering.labels_.size):
    print(clustering.labels_[x])'''

#print("Modularity: ", g.modularity(clustering.labels_,None))