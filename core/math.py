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
    if len(x) > 2:
        return x[0] * mu(x[1:])

    return x[0] * x[1]
