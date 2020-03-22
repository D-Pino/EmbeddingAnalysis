import torch as t
import pandas as pd
import numpy as np
from numpy.linalg import norm
import kmeans as k
import matplotlib.pyplot as plt

#load model into dictionary, access the embeddings, convert to numpy array
embeddings = t.load('facebook.pth')['embeddings'].numpy()






'''for i in range(2, 30):
    km = k.Kmeans(n_clusters=i, max_iter=100)
    km.fit(embeddings)
    print("Iteration " , i-1 , "complete.")
    distortions.append(km.error)
print(distortions)
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2,30), distortions, 'bo-')
plt.grid(True)
plt.title('Elbow curve')
plt.show()'''


