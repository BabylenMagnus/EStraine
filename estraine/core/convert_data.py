import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def class2numeric(data: pd.core.frame.DataFrame):
    """
    Convert feature to \mathbb{N}
    """
    for column in data.columns:
        original_value = list(data[column].drop_duplicates())
        new_value = dict(zip(original_value, range(len(original_value))))
        data[column] = data[column].map(new_value)
    return data


def norm(data: pd.core.frame.DataFrame):
    """
    Normalize data to:
        mean = 0; std = 1
    """
    for column in data.columns:
        mean = data[column].mean()
        std = data[column].std()

        if std == 0:
            continue

        new_data = np.array(data[column])
        data[column] = (new_data - mean) / std

    return data


def visualization(X, Y, s=15, alpha=.8):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    plt.figure(figsize=(20, 10))    # change size WxH
    plt.scatter(X[:, 0], X[:, 1],   # draw dots for 2 vector
                c=Y,                # colors
                alpha=alpha, s=s,   # alpha - transparent colors, s - size of dots
                cmap=plt.cm.get_cmap('gist_rainbow', len(np.unique(Y))))
    plt.colorbar()
    plt.show()
