from .QQTensor import QQBase
import numpy as np

class QQLoss(QQBase):
    """Abstract layer class.

    Parameters
    ----------
    name : str
        Tenser name.
    """

    def __init__(self, name):
        super(QQLoss, self).__init__(name)

    def __call__(self, t, p):
        if not self._initialized:
            self.init(p)
        self.x['t'] = t
        self.x['p'] = p

    def init(self, x):
        self._initialized = True
        self.params_pool = self.params_pool + x.params_pool

class QQMSE(QQLoss):
    """Mean squared error (MSE) Loss.

    Parameters
    ----------
    name : str, optional
        Tensor name.
    """
    
    def __init__(self, name = 'undefined'):
        super(QQMSE, self).__init__(name)

    def __call__(self, t, p):
        super(QQMSE, self).__call__(t, p)
        
        self.out = (self.x['t'].out-self.x['p'].out) ** 2
        return self
    
    def backward(self, partial=None):
        label = self.x['t'].out
        prediction = self.x['p'].out
        
        if partial:
            partial.grad_weight = -2 * (label-prediction) * (self.x['p'].backward(partial))
        else:
            for param in self.params_pool:
                param.grad_weight = -2 * (label-prediction) * (self.x['p'].backward(param))
        return None
