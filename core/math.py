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
    return n if n <= 1 else fib(n - 1) + fib(n - 2)
