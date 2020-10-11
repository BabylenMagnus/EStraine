import numpy as np


class LinearClassifier:
    def __init__(self, xl, yl):
        self.feat = xl.shape[1]
        self.weight = (np.random.rand(self.feat) / 2) - (1 / 4)
        self.l = xl.shape[0]
        self.pred = lambda x: (self.weight * x).sum()

    def margin(self, x, y):
        pred = self.pred(x)
        if y == 1:
            return abs(y - pred) if pred >= 0 else -abs(y - pred)
        else:
            return abs(y - pred) if pred <= 0 else -abs(y - pred)

    def stochastic_gradient(self):
        pass

    def predict(self, x):
        return int(self.pred(x) >= 0)
