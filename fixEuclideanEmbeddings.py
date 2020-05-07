#this file is to fix the format of euclidean embeddings when we get it from node2vec/deepwalk cause it's not usable right away
import pandas as pd
import numpy as np

#before this, replaced all the spaces with commas (idk ctrl+h i think) and removed the first two summary numbers of the file, then saved as csv
X = pd.read_csv('facebooknode2vecEmbeddings.csv', sep =',', header = None)
X = X.sort_values(by = 0)
X = X.reset_index()
X = X.drop(["index",0], axis = 1)
embeddings = X.to_numpy()

np.savetxt('facebooknode2vecEmbeddings.csv', embeddings, delimiter = ',')

