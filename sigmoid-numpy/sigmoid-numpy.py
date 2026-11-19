import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x = np.asarray(x, dtype=float)
    z = np.exp(-np.abs(x))
    result = np.where(x >= 0, 1 / (1 + z), z / (1 + z))
    return float(result) if result.ndim == 0 else result