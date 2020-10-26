from estraine.models.SVM import SVM
import pandas as pd
from estraine.core.convert_data import class2numeric, norm
import numpy as np


data = pd.read_csv("/data/mushrooms.csv")
data = class2numeric(data)
data['bias'] = 1
Y = data['class']
X = data.drop(['class'], axis=1)
X = norm(X)
Y = np.array(Y)
X = np.array(X)
Y = (Y * 2) - 1


def test_svm():
    acc = []
    for _ in range(10):
        model = SVM(X, Y)
        model.fit(epochs=1)
        acc.append(np.sum(model.predict(X) == Y) / len(X))
    assert min(acc) > .96
