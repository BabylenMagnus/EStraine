from core.math import mu


class NaiveBayes:
    """
    y = argmax_{y} P(y) \Pi^n_{i=1} P(xi|y)
    """
    def __init__(self, xl, yl):

        self.Xl = xl
        self.Yl = yl

        self.Xy0 = xl[yl == 0]
        self.Xy1 = xl[yl == 1]

        self.lenset0 = self.Xy0.shape[0]
        self.lenset1 = self.Xy1.shape[0]

    def predict(self, x):

        prob0, prob1 = list(), list()

        for i in range(self.Xy0.shape[1]):
            prob0.append(len(self.Xy0[self.Xy0[:, i] == x[i]]) / self.lenset0)

        for i in range(self.Xy1.shape[1]):
            prob1.append(len(self.Xy1[self.Xy1[:, i] == x[i]]) / self.lenset1)

        prob0, prob1 = mu(prob0), mu(prob1)

        return 0 if prob0 > prob1 else 1
