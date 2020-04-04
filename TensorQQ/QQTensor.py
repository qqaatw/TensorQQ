import numpy as np

class QQTensor(object):
    def __init__(self, unit, name):
        self._initialized = False

        self.name = name
        self.unit = unit

        self.x = dict()
        self.out = None
        self.weight = dict()
        self.bias = dict()
        self.params_pool = list()
     
    def __repr__(self):
        text = []
        text.append('Tensor Name: {}'.format(self.name))
        
        text.append('--------------------------------------')

        text.append('Weights:')
        for w in self.weight:
            text.append('{} =\n {}'.format(w, self.weight[w]))
        
        text.append('--------------------------------------')

        text.append('Bias:')
        for b in self.bias:
            text.append('{} =\n {}'.format(b, self.bias[b]))

        text.append('--------------------------------------')

        text.append('Out:')
        text.append(str(self.out))
        return '\n'.join(text)

    def __call__(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

class QQVariable(QQTensor):
    def __init__(self, array, name = 'undefined'):
        super(QQVariable, self).__init__(None, name)
        self.out = array
    def __call__(self):
        return self
    
    def backward(self):
        return None
    
    @property
    def shape(self):
        return self.out.shape

class QQParameter(QQTensor):
    def __init__(self, array=None, initializer=None, name='undefined'):
        super(QQParameter, self).__init__(None, name)
        self.initializer = initializer
        self.weight = np.array(array) if not isinstance(array, np.ndarray) else array
        self.grad_weight = None

    def __repr__(self):
        return str(self.weight)

    def __call__(self):
        return self
    
    def backward(self, partial):
        if partial is self:
            return 1
        else:
            return 0
    
    def init(self, shape, is_bias=False):
        self._initialized = True
        if is_bias:
            self.weight = np.ones(shape) * self.initializer(None)
        else:
            self.weight = self.initializer(shape)

    @property
    def shape(self):
        return self.weight.shape
