import numpy as np
from .QQOperator import sqrt

class QQOptimizer(object):
    def __init__(self, lr):
        self.lr = lr

    def update(self, loss, batch_size, partial=None):
        raise NotImplementedError

class QQSGD(QQOptimizer):
    def __init__(self, lr):
        super(QQSGD, self).__init__(lr)
    
    def update(self, loss, batch_size, partial=None):
        params = [partial] if partial else loss.params_pool
        for param in params:
            param.weight -= self.lr * param.grad_weight

class QQRMSProp(QQOptimizer):
    """http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    def __init__(self, lr, forgetting_fector=0.9, epsilon=1e-7):
        super(QQRMSProp, self).__init__(lr)
        self.v = dict()
        self.epsilon = epsilon
        self.forgetting_factor = forgetting_fector
    
    def update(self, loss, batch_size, partial=None):
        params = [partial] if partial else loss.params_pool
        for param in params:
            if param not in self.v:
                self.v[param] = 0.
            self.v[param] = self.forgetting_factor * self.v[param] + \
                (1 - self.forgetting_factor) * param.grad_weight
            param.weight -= (self.lr / sqrt(self.v[param] + self.epsilon)) * param.grad_weight

class QQAdam(QQOptimizer):
    """https://arxiv.org/abs/1412.6980
    """

    def __init__(self, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super(QQAdam, self).__init__(lr)
        self.m = dict()
        self.v = dict()
        self.t = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def update(self, loss, batch_size, partial=None):
        self.t += 1
        params = [partial] if partial else loss.params_pool
        for param in params:
            if param not in self.v:
                self.m[param] = 0.
                self.v[param] = 0.
            self.m[param] = self.beta_1 * self.m[param] + (1 - self.beta_1) * param.grad_weight   
            self.v[param] = self.beta_2 * self.v[param] + (1 - self.beta_2) * np.power(param.grad_weight, 2)
            
            m_prime = self.m[param] / (1 - np.power(self.beta_1, self.t))
            v_prime = self.v[param] / (1 - np.power(self.beta_2, self.t))

            param.weight -= (self.lr * m_prime) / (sqrt(v_prime + self.epsilon))
