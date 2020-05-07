from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community
import pickle as pik

embeddings = np.genfromtxt('facebook_poin_2_50_emb.csv', delimiter = ",")

distances = np.genfromtxt('facebook_poin_2_50_dist.csv', delimiter=',')

clustering = AgglomerativeClustering(n_clusters=12, affinity="precomputed", linkage="complete").fit(distances)


with open('facebook_edgelist.txt', 'rb') as f:
    edgeList = pik.load(f)

nG = nx.Graph()
nG.add_nodes_from(range(embeddings.shape[0]))
nG.add_edges_from(edgeList)

partition = dict(enumerate(clustering.labels_))

nxModularity =  community.modularity(partition,nG)

print("networkx modularity: ", nxModularity)


'''
with open('twitchRUEdgeList.txt', 'rb') as f:
    edgeList = pik.load(f)

nG = nx.Graph()
nG.add_nodes_from(range(embeddings.shape[0]))
nG.add_edges_from(edgeList)

modularities = []
for i in range(2,50):
    print("Iteration : ", i+1)
    clustering = AgglomerativeClustering(n_clusters = i,affinity = "precomputed", linkage = "complete").fit(distances)
    partition = dict(enumerate(clustering.labels_))
    modularities.append(community.modularity(partition,nG))


print(modularities)

print("# of clusters: ", modularities.index(max(modularities)) + 1)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2,50), modularities, 'bo-')
plt.grid(True)
plt.title('Some kinda curve')
plt.show()
'''