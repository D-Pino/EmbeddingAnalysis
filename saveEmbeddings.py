import numpy as np
import torch as t

embeddings = t.load('facebook_poin_2_250_raw.pth')['embeddings'].numpy()

np.savetxt('facebook_poin_2_250_emb.csv', embeddings,delimiter = ",")

