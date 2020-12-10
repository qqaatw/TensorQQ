import numpy as np

def sqrt(x):
    y = np.sqrt(x)
    y[np.isnan(y)] = 0.
    return y


def fc(x, weight, bias):
    return np.matmul(weight, x) + bias
