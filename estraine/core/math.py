import numpy as np


def mu(x):
    """
     _________________________
    < multiple list of number >
     -------------------------
            \   ^__^
             \  (oo)\_______
                (__)\       )\/\
                    ||----w |
                    ||     ||
    """
    return x[0] * mu(x[1:]) if len(x) > 2 else x[0] * x[1]


def derivative(x, f, h=1e-3):
    """
    Search derivative f function in x with h scope
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def fib(n):
    a, b = 1, 1
    point = 2
    while point < n:
        point += 1
        a, b = b, a + b
    return b


def high_divisor(a, b):
    min_ = min(a, b)
    if not min_:
        return 0
    r = max(a, b) % min_
    if not r:
        return min_
    return high_divisor(r, min_)


def weight_initialization(n_features, dist=0.5):
    """
    initializes weights between -dist and dist
    """
    rand = np.random.rand(n_features)
    return rand * dist * 2 - dist
