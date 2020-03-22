import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#convert embeddings to necessary format, hoping to move to another file
X = pd.read_csv('facebooknode2vecEmbeddings.csv',sep = ',', header = None)
X = X.sort_values(by = 0)
X = X.reset_index()
X = X.drop(["index",0], axis = 1)
embeddings = X.to_numpy()
#print(embeddings)

distortions = []
for k in range(2, 100):
    print("Iteration: ", k-1)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings)
    distortions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 100), distortions, 'bo-')
plt.grid(True)
plt.title('Elbow curve')
plt.show()