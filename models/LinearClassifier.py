import numpy as np
import random


class LinearClassifier:
    def __init__(self, xl, yl):
        self.feat = xl.shape[1]
        self.weight = (np.random.rand(self.feat) / 2) - (1 / 4)
        self.l = xl.shape[0]
        self.pred = lambda x: (self.weight * x).sum()
        self.pred_ = lambda x: (self.weight * x).sum(axis=1)
        self.error_func = lambda x: x ** 2 if x < 0 else 0
        self.marg = self.margin_(xl, yl)
        self.Q = np.array(list(map(self.error_func, self.marg))).sum()

    def margin(self, precedent):
        x, y = precedent
        pred = self.pred(x)
        if y == 1:
            return abs(y - pred) if pred >= 0 else -abs(y - pred)
        else:
            return abs(y - pred) if pred <= 0 else -abs(y - pred)

    def margin_(self, xl, yl):
        margin = list(map(self.margin, zip(xl, yl)))
        return np.array(margin)

    def derivative(self, x, y, h=1e-5):
        """
        derivative fi(x) for chain rule
        """
        df = np.ones(self.feat)
        for i in range(self.feat):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] = x[i] + h
            x2[i] = x[i] - h

            df[i] = (self.pred(x1) - self.pred(x2)) / (2 * h)

        x_ = self.pred(x)
        dm = (self.margin((x_ + h, y)) - self.margin((x_ - h, y))) / (2 * h)

        x = self.margin((x, 0))
        dl = (self.error_func(x + h) - self.error_func(x - h)) / (2 * h)

        return dl * dm * df

    def stochastic_gradient(self, x, y, n=1e-3):
        i = random.choice(range(len(x)))
        dev = self.derivative(x[i], y[i])
        self.weight = "Думай покачто"

    def predict(self, x):
        return int(self.pred(x) >= 0)

    def predict_(self, x):
        return (self.pred_(x) >= 0).astype(int)
