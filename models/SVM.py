import numpy as np


class SVM:
    def __init__(self, xl, yl):
        self.Xl = xl
        self.Yl = yl

        self.obj = len(self.Xl)
        self.feat = self.Xl.shape[1]
        self.weight = (np.random.rand(self.feat) / 2) - (1 / 4)

        self.predict_ = lambda x: 1 if x > 0 else -1
        self.margin = lambda x, y: y * np.dot(self.weight, x)

    def margin_(self):
        marg = []

        for x, y in zip(self.Xl, self.Yl):
            marg.append(self.margin(x, y))

        return np.array(marg)

    def predict(self, x):
        pred = map(self.predict_, (x * self.weight).sum(axis=1))
        return np.array(list(pred))

    def fit(self, epochs=100, n=1e-3, c=1e-3, m=12):
        for epoch in range(int(epochs * self.obj)):
            i = np.random.randint(self.obj)
            x, y = self.Xl[i], self.Yl[i]

            margin = self.margin(x, y)

            if margin >= 1:
                self.weight -= n / max((epoch // m), 1) * c * self.weight
            else:
                self.weight -= (n / max((epoch // m), 1) * c * self.weight) - x * y

        return self
