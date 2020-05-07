import numpy as np
import helpers as mine


embeddings = np.genfromtxt('facebook_poin_2_250_emb.csv' , delimiter = ",")

distances = np.zeros((embeddings.shape[0],embeddings.shape[0]))
for x in range(distances.shape[0]):
    for y in range(distances.shape[0]):
        distances[x,y] = mine.poincareDistancePoint(embeddings[x,:],embeddings[y,:])

np.savetxt('facebook_poin_2_250_dist.csv', distances, delimiter = ",")