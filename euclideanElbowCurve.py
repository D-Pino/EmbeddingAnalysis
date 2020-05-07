import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

embeddings = np.genfromtxt('facebooknode2vecEmbeddings.csv', delimiter = ",")

distortions = []
for k in range(2, 200):
    print("Iteration: ", k-1)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings)
    distortions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 200), distortions, 'bo-')
plt.grid(True)
plt.title('Elbow curve')
plt.show()