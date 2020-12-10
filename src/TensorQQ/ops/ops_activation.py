import numpy as np

from .QQOperator import QQOperator

__all__ = ['Sigmoid', 'ReLU']

class ReLU(QQOperator):
    @classmethod
    def forward(cls, *args):
        super().forward(*args)
        return np.maximum(np.zeros(args[0].shape), args[0].v)
    
    @classmethod
    def backward(cls, partial, self_tensor):
        super().backward(partial, self_tensor)
        step = np.copy(self_tensor.parents[0].v)
        step[step > 0] = 1
        step[step <= 0] = 0
        rt = self_tensor.parents[0]._backward(partial)
        rt._gradient_internal['count'] = 0
        rt.grad = rt.grad.dot(step)
        # If partial is a bias, the gradient will be flattened,
        # otherwise it will transpose step (relu's own derivatives) first, then calculte the gradient.
        return rt


class Sigmoid(QQOperator):
    @classmethod
    def forward(cls, *args):
        super().forward(*args)
        return 1 / (1 + np.exp(-args[0].v))

    @classmethod
    def backward(cls, partial, self_tensor):
        super().backward(partial, self_tensor)
        return np.multiply(np.mean((self_tensor.parents[0].v * (1 - self_tensor.parents[0].v)), axis=0, keepdims=True).T, parents[0].backward(partial))
