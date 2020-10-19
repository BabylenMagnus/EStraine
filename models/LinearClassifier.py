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
        return pred * y

    def margin_(self, xl, yl):
        margin = list(map(self.margin, zip(xl, yl)))
        return np.array(margin)

    def derivative(self, x, y, h=1e-5):
        """
        derivative fi(x) for chain rule
        """
        df = np.ones(self.feat)
        for i in range(self.feat):

            weight1 = self.weight.copy()
            weight2 = self.weight.copy()

            weight1[i] = self.weight[i] + h
            weight2[i] = self.weight[i] - h

            df[i] = (self.pred(weight1) - self.pred(weight2)) / (2 * h)

        x_ = self.pred(x)
        dm = (self.margin((x_ + h, y)) - self.margin((x_ - h, y))) / (2 * h)

        x = self.margin((x, 0))
        dl = (self.error_func(x + h) - self.error_func(x - h)) / (2 * h)

        return dl * dm * df

    def stochastic_gradient(self, x, y, n=1e-2):
        i = random.choice(range(len(x)))
        dev = self.derivative(x[i], y[i])
        self.weight = self.weight - n * dev

    def predict(self, x):
        return int(self.pred(x) >= 0)

    def predict_(self, x):
        return (self.pred_(x) >= 0).astype(int)
