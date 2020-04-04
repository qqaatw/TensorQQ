from .QQTensor import QQTensor
from .QQTensor import QQParameter
import numpy as np


class QQLayer(QQTensor):
    def __init__(self, unit, initializer=None, name='undefined'):
        super(QQLayer, self).__init__(unit, name)
        self.initializer = initializer
    
    def __call__(self, x):
        if not self._initialized:
            self.init(x)
        self.x['x'] = x

    def init(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.out.shape

class QQDense(QQLayer):
    def __init__(self, unit, weight=None, bias=None, initializer=None, name='undefined'):
        super(QQDense, self).__init__(unit, initializer, name)
        self.weight['weight_1'] = weight
        self.bias['bias_1'] = bias

    def __call__(self, x):
        super(QQDense, self).__call__(x)
        self.out = np.matmul(
            self.weight['weight_1'].weight, x.out) + self.bias['bias_1'].weight
        return self

    def backward(self, partial=None):
        if partial is self.weight['weight_1']:
            return self.x['x'].out.T  # should be transposed.
        elif partial is self.bias['bias_1']:
            return 1
        else:
            return np.dot(self.weight['weight_1'].weight, self.x['x'].backward(partial))

    def init(self, x):
        self._initialized = True
        if self.initializer:
            if self.weight['weight_1']:
                assert self.weight['weight_1'].shape == (self.unit, x.shape[0]),\
                    '{}\'s weight has an invalid shape.'.format(self.name)
            else:
                self.weight['weight_1'] = QQParameter(
                    initializer=self.initializer, name=self.name + '_weight')
                self.weight['weight_1'].init((self.unit, x.shape[0]), is_bias=False)
            if self.bias['bias_1']:
                assert self.bias['bias_1'].shape == (self.unit, 1),\
                    '{}\'s bias has an invalid shape.'.format(self.name)
            else:
                self.bias['bias_1'] = QQParameter(
                    initializer=self.initializer, name=self.name + '_bias')
                self.bias['bias_1'].init((self.unit, 1), is_bias=True)
        elif self.weight['weight_1'] is None or self.bias['bias_1'] is None:
            raise ValueError('{}\'s weight or bias has not been initialized!'.format(self.name))

        if not isinstance(self.weight['weight_1'], QQParameter)\
            or not isinstance(self.bias['bias_1'], QQParameter):
            raise TypeError('{}\'s weight or bias is not a QQparameter instance'.format(self.name))

        # Collect parameters
        self.params_pool.append(self.weight['weight_1'])
        self.params_pool.append(self.bias['bias_1'])
        self.params_pool = self.params_pool + x.params_pool
