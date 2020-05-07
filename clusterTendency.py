import pyclustertend as pyc
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X,y = make_blobs(n_samples = 4000, centers = 12, n_features = 5)

eucEmbeddings = np.genfromtxt('facebooknode2vecEmbeddings.csv', delimiter = ",")

hypEmbeddings = np.genfromtxt('facebookPoincareEmbeddings.csv', delimiter = ",")

eucStat = pyc.hopkins(eucEmbeddings, 150)

hypStat = pyc.hopkins(hypEmbeddings, 150)

xStat = pyc.hopkins(X, 150)

print("Hopkins Statistic for euclidean embeddings: ", eucStat )

print("Hopkins Statistic for hyperbolic poincare embeddings: ", hypStat )

print("Hopkins Statistic for fake points: ", xStat )

pyc.ivat(X)
print("hello there")



'''eucScore =pyc.assess_tendency_by_metrics(eucEmbeddings)
hypScore =pyc.assess_tendency_by_metrics(hypEmbeddings)
xScore =pyc.assess_tendency_by_metrics(X)

print('eucScore: ', eucScore)
print('hypScore: ', hypScore)
print('xScore: ', xScore)'''

