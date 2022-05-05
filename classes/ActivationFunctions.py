import numpy as np
from numba import jit, cuda

def logistic_f(x: np.ndarray) -> np.ndarray:
        f = lambda x: 1.0 / (1.0 + np.exp(-x))
        return f(x)

def logistic_d(x: np.ndarray) -> np.ndarray:
        return logistic_f(x) * (1.0 - logistic_f(x))

def relu_f(x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

def relu_d(x: np.ndarray) -> np.ndarray:
        return 1.0 * (x > 0)
