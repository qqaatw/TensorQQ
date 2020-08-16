# TODO: 

import warnings

import numpy as np

from TensorQQ.QQOperator import auto_forward, auto_backward, ops_list, shared_mul_div
from TensorQQ.QQInitializer import QQInitializer

class QQBase(object):
    """Abstract base class, providing tensor attributes, representaiton and functions. 

    Parameters
    ----------
    name : str
        Tensor name.
    """

    def __init__(self, name):
        self._initialized = False

        self.name = name

        self.x = dict()
        self.weight = dict()
        self.bias = dict()
        self.params_pool = list()
     
    def __repr__(self):
        text = []

        text.append('======================================')
        text.append('Tensor Name: {}'.format(self.name))
        text.append('======================================')

        text.append('Weights:')
        for w in self.weight:
            text.append('{} =\n {}'.format(w, self.weight[w]))
        
        text.append('--------------------------------------')

        text.append('Bias:')
        for b in self.bias:
            text.append('{} =\n {}'.format(b, self.bias[b]))

        text.append('--------------------------------------')

        text.append('Output:')
        text.append(str(self.out))

        return '\n'.join(text)

    def __call__(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

class QQVariable(QQBase):
    """Storing constant variable.

    Parameters
    ----------
    array : array-like
        Variable array.
    name : str, optional
        Tensor name, by default 'undefined'.
    """

    def __init__(self, array, name = 'undefined'):
        super(QQVariable, self).__init__(name)
        self.out = np.array(array) if not isinstance(array, np.ndarray) else array
    
    def __call__(self):
        return self
    
    def backward(self):
        return None
    
    @property
    def shape(self):
        return self.out.shape

class QQParameter(QQBase):
    """Storing differentiable parameter.

    Parameters
    ----------
    array : array-like, optional
        Parameter array. if not given, initializer should be specified, by default None.
    initializer : QQInitializer, optional
        Initializer, by default None.
    name : str, optional
        Tensor name, by default 'undefined'.
    """

    def __init__(self, array=None, initializer=None, name='undefined'):
        super(QQParameter, self).__init__(name)
        if array is None:
            assert initializer is not None, \
                "If array is not given, initializer should be specified."
            assert isinstance(initializer, QQInitializer), \
                "The initializer should be an instance of QQInitializer."
        else:
            self._initialized = True
        
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
        if self._initialized:
            msg = "Parameter '{}' has been initialized.".format(self.name)
            warnings.warn(msg)
        if is_bias:
            self.weight = np.ones(shape) * self.initializer()
        else:
            self.weight = self.initializer(shape)
        
        self._initialized = True

    @property
    def shape(self):
        return self.weight.shape

class QQTensor(object):
    """Tensor object used by TensorQQ

    Parameters
    ----------
    v : float
        Computed tensor value.
    op : str, optional
        Operator name, by default None.
    parent : Tuple of QQTensors, optional
        (Parent1, Parent2), by default None.
    differentiable : bool, optional
        Whether this tensor is differentiable, by default True.
    name : str, optional
        Tensor name, by default "undefined".
    """
    
    # Basic Operators
    # param : parameter itself
    # add   : addition
    # sub   : subtration
    # mul   : multiplicaiton
    # div   : division
    basic_op = ['param', 'add', 'sub', 'mul', 'div']
    

    def __init__(self, v, op='param', parents=None, differentiable=True, name='undefined'):
        assert isinstance(parents, tuple) or parents is None, \
            '`parent` should be None or a tuple. : {}'.format(parents)
        if op == 'param':
            self._params_pool = [self]
        else:
            self._params_pool = []
            for parent in parents:
                self._params_pool.extend(parent.params)

        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype='float32')

        self._value = v
        self._gradient = None
        self._operator = op
        self._parent = parents
        self._differentiable = differentiable
        self._name = name

    # Operators
    @classmethod
    def _op(cls, op_name, *args, name=None):
        assert all([isinstance(arg, QQTensor) for arg in args]), \
            "Arguments of operator should be an instance of `QQTensor` : {}".format([type(arg.v) for arg in args])

        if name is None:
            name = '{}_{}'.format(op_name, '_'.join([arg.name for arg in args]))

        if op_name == 'add':
            return cls(args[0].v + args[1].v, op_name, (args[0], args[1]), name=name)
        elif op_name == 'sub':
            return cls(args[0].v - args[1].v, op_name, (args[0], args[1]), name=name)
        elif op_name == 'mul':
            return cls(args[0].v * args[1].v, op_name, (args[0], args[1]), name=name)
        elif op_name == 'div':
            return cls(args[0].v / args[1].v, op_name, (args[0], args[1]), name=name)
        else:
            return cls(auto_forward(op_name, *args), op_name, args, name=name)

    def __add__(self, other):
        return self._op('add', self, other)

    def __sub__(self, other):
        return self._op('sub', self, other)

    def __mul__(self, other):
        return self._op('mul', self, other)
    
    def __div__(self, other):
        return self._op('div', self, other)
    
    def __iadd__(self, other):
        return self._op('add', self, other)
    
    def __isub__(self, other):
        return self._op('sub', self, other)

    def __imul__(self, other):
        return self._op('mul', self, other)
    
    def __idiv__(self, other):
        return self._op('div', self, other)

    @staticmethod
    def matmul(x1, x2):
        return QQTensor._op('matmul', x1, x2)

    # End of Operators

    @property
    def T(self):
        """Transpose value.

        Returns
        -------
        array-like
            Transposed value.
        """
        
        self._value = self._value.T
        return self

    @property
    def v(self):
        return self._value
    
    @property
    def grad(self):
        return self._gradient

    @property
    def name(self):
        return self._name
    
    @property
    def params(self):
        return self._params_pool

    @property
    def shape(self):
        return self._value.shape

    def backward(self, partial=None):
        if partial is None:
            for param in self._params_pool:
                param._gradient = self._backward(param)
            return None
        else:
            partial._gradient = self._backward(partial)
            return partial._gradient

    def _backward(self, partial):
        if not self._differentiable:
            return 0
        elif self._differentiable and self is partial:
            return np.ones(self.shape, dtype='float32')

        if self._operator in QQTensor.basic_op:
            if self._operator == 'param':
                return 0
            elif self._operator == 'add':
                return self._parent[0]._backward(partial) + self._parent[1]._backward(partial)
            elif self._operator == 'sub':
                return self._parent[0]._backward(partial) - self._parent[1]._backward(partial)
            elif self._operator == 'mul':
                grad_p0, grad_p1 = shared_mul_div(partial, self._parent[0], self._parent[1])
                return grad_p0 * grad_p1
            elif self._operator == 'div':
                grad_p0, grad_p1 = shared_mul_div(partial, self._parent[0], self._parent[1])
                return grad_p0 / grad_p1
        else:
            return auto_backward(self._operator, partial, *self._parent)


class QQTrainer(object):
    pass

class QQFullyConnection(object):
    def __init__(self, weight : QQTensor, bias : QQTensor, *args):
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        return QQTensor.matmul(x, self.weight.T) + self.bias


def test_QQTenser():
    weight = QQTensor(np.random.normal(size=(5, 2)))
    bias = QQTensor(np.random.normal(size=(5, 1)))
    X = QQTensor(np.random.normal(size=(2, 1)), differentiable=False)
    fc = QQFullyConnection(weight, bias)
    return fc, fc(X), X

if __name__ == "__main__":
    pass
