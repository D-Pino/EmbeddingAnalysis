import hdbscan
import numpy as np
import networkx as nx
import torch as t
import community
import pandas as pd


poincareEmbeddings = t.load('facebook.pth')['embeddings'].numpy()
lorentzEmbeddings = t.load('facebookLORENTZ.pth')['embeddings'].numpy()

clusterer = hdbscan.HDBSCAN(metric = 'precomputed')

distances = np.genfromtxt('lorentzDistanceMatrix.csv', delimiter=',')

clusterer.fit(distances)


df = pd.read_csv('facebook.csv')
df = df.drop('weight', axis = 1)

edgeList = []

for index, row in df.iterrows():
    edgeList.append((row[0],row[1]))

nG = nx.Graph()
nG.add_nodes_from(range(lorentzEmbeddings.shape[0]))
nG.add_edges_from(edgeList)

partition = dict(enumerate(clusterer.labels_))

nxModularity = community.modularity(partition,nG)

print("networkx modularity: ", nxModularity)

print(clusterer.labels_)
print(clusterer.labels_.max())
print(clusterer.probabilities_)
count = 0
for x in clusterer.labels_:
    if(x==-1):
        count+=1
print("Number of noise points: ", count)



