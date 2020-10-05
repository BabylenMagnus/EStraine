from core.math import multiple


class NaiveBayes:
    def __init__(self, Xl, Yl):

        self.Xl = Xl
        self.Yl = Yl

        self.Xy0 = Xl[Yl == 0]
        self.Xy1 = Xl[Yl == 1]

        self.denom0 = self.Xy0.shape[0]
        self.denom1 = self.Xy1.shape[0]

    def predict(self, x):

        prob0 = list()
        prob1 = list()

        for i in range(self.Xy0.shape[1]):
            prob0.append(len(self.Xy0[self.Xy0[:, i] == x[i]]) / self.denom0)

        for i in range(self.Xy1.shape[1]):
            prob1.append(len(self.Xy1[self.Xy1[:, i] == x[i]]) / self.denom1)

        prob0, prob1 = multiple(prob0), multiple(prob1)

        if prob0 > prob1:
            return 0
        else:
            return 1
