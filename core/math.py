def multiple(x):
    """
    multiple list of number
    """
    if len(x) > 2:
        return x[0] * multiple(x[1:])

    return x[0] * x[1]
