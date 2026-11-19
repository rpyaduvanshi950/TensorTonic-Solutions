import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        error = p - y

        w -= lr * (X.T @ error) / m
        b -= lr * np.mean(error)

    return w, float(b)