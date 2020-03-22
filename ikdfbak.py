import torch as t
import numpy as np

embeddings = t.load('facebook.pth')['embeddings'].numpy()

np.savetxt('bluhfalix.csv', embeddings, delimiter = ',')
