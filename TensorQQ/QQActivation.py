from TensorQQ.QQTensor import QQTensor
import numpy as np

class QQActivation(QQTensor):
    def __init__(self, name='undefined'):
        super(QQActivation, self).__init__(None, name)
    
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

class QQReLU(QQActivation):
    def __init__(self, name = 'undefined'):
        super(QQReLU, self).__init__(name)

    def __call__(self, x):
        super(QQReLU, self).__call__(x)
        self.out = np.maximum(np.zeros(x.out.shape), x.out)
        return self
    
    def backward(self, partial=None):
        x_out = np.copy(self.x['x'].out)
        x_out[x_out > 0] = 1
        x_out[x_out <= 0] = 0 
        return np.dot(x_out, self.x['x'].backward(partial))


class QQSigmoid(QQActivation):
    def __init__(self, name='undefined'):
        super(QQSigmoid, self).__init__(name)

    def __call__(self, x):
        super(QQSigmoid, self).__call__(x)
        self.out = 1 / (1 + np.exp(-x.out))
        return self

    def backward(self, partial=None):
        return np.dot((self.out * (1 - self.out)), self.x['x'].backward(partial))


class QQTanh(QQActivation):
    def __init__(self, name='undefined'):
        super(QQTanh, self).__init__(name)

    def __call__(self, x):
        super(QQTanh, self).__call__(x)
        self.out = (np.exp(x.out) - np.exp(-x.out)) / (np.exp(x.out) + np.exp(-x.out))
        return self

    def backward(self, partial=None):
        return 1 - np.power(self.out, 2)
