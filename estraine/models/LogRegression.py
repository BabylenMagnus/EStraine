import numpy as np
from estraine.core.math import weight_initialization


class LinearClassifier:
    def __init__(self, xl, yl):
        self.Xl, self.Yl = xl, yl
        self.n_samples, self.n_features = self.Xl.shape
        self.weight = weight_initialization(self.n_features)

        self.pred = lambda x: x.dot(self.weight)  # tests
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
        df = np.ones(self.n_features)
        for i in range(self.n_features):

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

    def stochastic_gradient(self, n=1e-2):
        i = np.random.randint(self.n_samples)
        dev = self.derivative(self.Xl[i], self.Yl[i])
        self.weight = self.weight - n * dev

    def predict(self, x):
        return int(self.pred(x) >= 0)

    def predict_(self, x):
        return (self.pred_(x) >= 0).astype(int)
