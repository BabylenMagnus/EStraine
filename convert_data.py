import pandas as pd


def class2numeric(data: pd.core.frame.DataFrame):
    """
    Convert feature to \mathbb{N}
    """
    for column in data.columns:
        original_value = list(data[column].drop_duplicates())
        new_value = dict(zip(original_value, range(len(original_value))))
        data[column] = data[column].map(new_value)
    return data
