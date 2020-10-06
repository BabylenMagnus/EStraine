import pandas as pd
import numpy as np


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
