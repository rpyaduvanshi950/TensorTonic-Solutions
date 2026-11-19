import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    b = []
    for a in x:
        if a > 0:
            b.append(a)
        else:
            b.append(alpha * (math.exp(a) - 1))
    return b