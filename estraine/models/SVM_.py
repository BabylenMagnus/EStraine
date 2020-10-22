import numpy as np
from estraine.core.math import weight_initialization


class Polynomial():
    def __call__(self, *args, **kwargs):
        pass


class SVM_:
    def __init__(self, xl, yl):

        self.Xl, self.Yl = xl, yl

        self.n_samples, self.n_features = self.Xl.shape
        self.Lambda = weight_initialization(self.n_features)

        self.pred = lambda x: (self.Lambda * x).sum()
        self.pred_ = lambda x: (self.Lambda * x).sum(axis=1)
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

    def fit(self, epochs=100, n=1e-3, c=1e-3, m=12):
        for epoch in range(int(epochs * self.n_samples)):
            i = np.random.randint(self.n_samples)
            x, y = self.Xl[i], self.Yl[i]

            margin = self.margin(x, y)

            if margin >= 1:
                self.Lambda -= n / max((epoch // m), 1) * c * self.Lambda
            else:
                self.Lambda -= (n / max((epoch // m), 1) * c * self.Lambda) - x * y

        return self
