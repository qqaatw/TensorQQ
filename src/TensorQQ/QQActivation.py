from .QQTensor import QQBase, QQTensor
import numpy as np


class QQActivation:
    """Abstract activation class.

    Parameters
    ----------
    name : str
        Tensor name.
    """

    def __init__(self, name):
        self._name = name

    def __call__(self, op, x):
        return QQTensor._op(op, x, name=self._name)


class QQSigmoid(QQActivation):
    """Sigmoid activation, also called logistic function.

    Parameters
    ----------
    name : str, optional
        Tensor name.
    """

    def __init__(self, name=None):
        super().__init__(name)

    def __call__(self, x):
        return super().__call__('sigmoid', x)


class QQReLU(QQActivation):
    """Rectified Linear Unit (ReLU) activation.

    Parameters
    ----------
    name : str, optional
        Tensor name.
    """

    def __init__(self, name=None):
        super().__init__(name)

    def __call__(self, x):
        return super().__call__('relu', x)

class QQActivation_deprecated(QQBase):
    """Abstract activation class.

    Parameters
    ----------
    name : str
        Tenser name.
    """

    def __init__(self, name='undefined'):
        super().__init__(name)
    
    def __call__(self, x):
        if not self._initialized:
            self.init(x)
        self.x['x'] = x
    
    def init(self, x):
        self._initialized = True
        self.params_pool = self.params_pool + x.params_pool

    @property
    def shape(self):
        return self.out.shape


class QQReLU_deprecated(QQActivation_deprecated):
    """Rectified Linear Unit (ReLU) activation.

    Parameters
    ----------
    name : str, optional
        Tensor name.
    """

    def __init__(self, name = 'undefined'):
        super().__init__(name)

    def __call__(self, x):
        super().__call__(x)
        self.out = np.maximum(np.zeros(x.out.shape), x.out)
        return self
    
    def backward(self, partial=None):
        x_out = np.copy(self.x['x'].out)
        x_out[x_out > 0] = 1
        x_out[x_out <= 0] = 0 
        return np.dot(x_out, self.x['x'].backward(partial))


class QQSigmoid_deprecated(QQActivation_deprecated):
    """Sigmoid activation, also called logistic function.

    Parameters
    ----------
    name : str, optional
        Tensor name.
    """

    def __init__(self, name='undefined'):
        super().__init__(name)

    def __call__(self, x):
        super().__call__(x)
        self.out = 1 / (1 + np.exp(-x.out))
        return self

    def backward(self, partial=None):
        return np.dot((self.out * (1 - self.out)), self.x['x'].backward(partial))


class QQTanh_deprecated(QQActivation_deprecated):
    """Hyperbolic tangent (Tanh) activation.

    Parameters
    ----------
    name : str, optional
        Tensor name.
    """

    def __init__(self, name='undefined'):
        super().__init__(name)

    def __call__(self, x):
        super().__call__(x)
        self.out = (np.exp(x.out) - np.exp(-x.out)) / (np.exp(x.out) + np.exp(-x.out))
        return self

    def backward(self, partial=None):
        return np.dot(1 - np.power(self.out, 2), self.x['x'].backward(partial))
