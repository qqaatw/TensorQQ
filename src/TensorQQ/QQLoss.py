from TensorQQ.QQTensor import QQTensor
import numpy as np

class QQMSE(QQTensor):
    def __init__(self, name = 'undefined'):
        super(QQMSE, self).__init__(None, name)

    def __call__(self, t, p):
        if not self._initialized:
            self.init(p)
        self.x['t'] = t
        self.x['p'] = p
        self.out = (t.out-p.out) * (t.out-p.out)
        return self
    
    def backward(self, partial=None):
        x_t_out = self.x['t'].out
        x_p_out = self.x['p'].out
        if partial:
            partial.grad_weight = -2 * (x_t_out-x_p_out) * (self.x['p'].backward(partial))
        else:
            for param in self.params_pool:
                param.grad_weight = -2 * (x_t_out-x_p_out) * (self.x['p'].backward(param))
        return None
    
    def init(self, x):
        self._initialized = True
        self.params_pool = self.params_pool + x.params_pool
