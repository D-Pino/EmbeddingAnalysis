import hdbscan
import torch as t
import pandas as pd
import numpy as np
import networkx as nx
import community


X = pd.read_csv('facebookDeepwalkEmbeddings.csv', sep =',', header = None)
X = X.sort_values(by = 0)
X = X.reset_index()
X = X.drop(["index",0], axis = 1)
embeddings = X.to_numpy()

clusterer = hdbscan.HDBSCAN()
clusterer.fit(embeddings)

df = pd.read_csv('facebook.csv')
df = df.drop('weight', axis = 1)

edgeList = []

for index, row in df.iterrows():
    edgeList.append((row[0],row[1]))

nG = nx.Graph()
nG.add_nodes_from(range(embeddings.shape[0]))
nG.add_edges_from(edgeList)

partition = dict(enumerate(clusterer.labels_))

nxModularity = community.modularity(partition,nG)

print("networkx modularity: ", nxModularity)

'''print(clusterer.labels_)
print(clusterer.labels_.max())
print(clusterer.probabilities_)
count = 0
for x in clusterer.labels_:
    if(x==-1):
        count+=1
print("Number of noise points: ", count)'''








