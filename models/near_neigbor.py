import numpy as np


def p(a, b):
    x = a - b
    dist = np.sum(x ** 2)
    return np.sqrt(dist)


class NearNeighbor:
    def __init__(self, Xl, Yl):
        self.Xl = Xl
        self.Yl = Yl

    def predict(self, x):
        f = lambda n: p(x, n)
        dist = np.array(list(map(f, self.Xl)))
        return self.Yl[np.argmin(dist)]
