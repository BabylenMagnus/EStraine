import numpy as np


def p(a, b):
    x = a - b
    dist = np.sum(x ** 2)
    return np.sqrt(dist)


class NearNeighbor:
    def __init__(self, xl, yl):
        self.Xl = xl
        self.Yl = yl

    def predict(self, x):
        dist = list(map(lambda n: p(x, n), self.Xl))
        dist = np.array(dist)
        return self.Yl[np.argmin(dist)]
