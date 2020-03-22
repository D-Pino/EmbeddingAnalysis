import numpy as np
from numpy.linalg import norm

def poincareDistancePoint(x, y):
    return np.arccosh(1 + 2 * (np.square(norm(x - y)) / ((1 - np.square(norm(x))) * (1 - np.square(norm(y))))))


def lorentzInnerProduct(x,y):
    return np.dot(x,y)-2*(x[0]*y[0])


def lorentzDistancePoint(x,y):
    clippedProd =  np.clip(-1*(lorentzInnerProduct(x,y)),1 ,None)
    return np.arccosh(clippedProd)

