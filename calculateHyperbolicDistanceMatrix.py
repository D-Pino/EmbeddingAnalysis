#this file is one use only, distance matrix is saved thereafter
import torch as t
import numpy as np
from numpy import savetxt
import helpers as mine


#load model into dictionary, access the embeddings, convert to numpy array
embeddings = t.load('facebookLORENTZ.pth')['embeddings'].numpy()


distances = np.zeros((embeddings.shape[0],embeddings.shape[0]))
for x in range(distances.shape[0]):
    for y in range(distances.shape[0]):
        distances[x,y] = mine.lorentzDistancePoint(embeddings[x,:],embeddings[y,:])

np.savetxt('lorentzDistanceMatrix.csv', distances, delimiter = ',')