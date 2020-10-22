from estraine.core.math import mu
import numpy as np


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


class NaiveBayes_:
    def __init__(self, xl, yl):

        self.Xl = xl

        self.classes = np.sort(np.unique(yl))
        self.class_count = np.zeros(len(self.classes), dtype=np.float64)

        self.Y = yl.reshape((-1, 1))
        self.Y = np.concatenate((1 - self.Y, self.Y), axis=1)
        self.class_count += self.Y.sum(axis=0)

        self.n_feature = xl.shape[1]
        self.category_count = self.count()
        self.feature_prob = self.update_feature_prob()

    def count(self):
        def _update_cat_count_dims(cat_count, highest_feature):
            diff = highest_feature + 1 - cat_count.shape[1]
            if diff > 0:
                # we append a column full of zeros for each new category
                return np.pad(cat_count, [(0, 0), (0, diff)], 'constant')
            return cat_count

        def _update_cat_count(X_feature, Y, cat_count, n_classes):
            for j in range(n_classes):
                mask = Y[:, j].astype(bool)
                counts = np.bincount(X_feature[mask], weights=None)
                indices = np.nonzero(counts)[0]
                cat_count[j, indices] += counts[indices]

        n_categories_ = self.Xl.max(axis=0)
        category_count = [np.zeros((len(self.classes), 0)) for _ in range(self.n_feature)]

        for i in range(self.n_feature):
            x_feature = self.Xl[:, i]
            category_count[i] = _update_cat_count_dims(category_count[i],
                                                       n_categories_[i])
            _update_cat_count(x_feature, self.Y,
                              category_count[i],
                              self.class_count.shape[0])

        return category_count

    def update_feature_prob(self):
        feature_log_prob = []
        for i in range(self.n_feature):
            smoothed_cat_count = self.category_count[i]
            feature_log_prob.append(np.array([i / i.sum() for i in smoothed_cat_count]))
        return feature_log_prob

    def predict(self, x):

        jll = np.zeros((1, len(self.classes)))

        for i in range(self.n_feature):
            indices = x[i]
            jll += self.feature_prob[i][:, indices]

        return np.argmax(jll, axis=1).item()