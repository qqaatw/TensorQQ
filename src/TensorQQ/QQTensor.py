# TODO: 

import warnings

import numpy as np

from .ops.QQOperator import auto_forward, auto_backward
from .QQInitializer import QQInitializer
from .QQSetting import *

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

class QQTensor:
    """Tensor object used by TensorQQ.

    Parameters
    ----------
    v : float
        Computed tensor value.
    op : str, optional
        Operator name, by default "param".
    parent : Tuple of QQTensors, optional
        (Parent1, Parent2), by default None.
    differentiable : bool, optional
        Whether this tensor is differentiable, by default True.
    name : str, optional
        Tensor name, by default "undefined".
    """

    def __init__(self, v, op='param', parents=None, differentiable=True, name='undefined'):
        assert isinstance(parents, tuple) or parents is None, \
            '`parent` should be None or a tuple. : {}'.format(parents)
        if op == 'param':
            self._params_pool = [self]
        else:
            self._params_pool = []
            for parent in parents:
                self._params_pool.extend(parent.params)

        self._differentiable = differentiable
        self._gradient = None
        self._gradient_internal = None
        self._name = name
        self._operator = op
        self._parent = parents
        self._value = np.array(v, dtype=TENSORQQ_DTYPE) if not isinstance(v, np.ndarray) else v
        self.reset_internal()

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

    def __matmul__(self, other):
        return self._op('matmul', self, other)

    # Operators

    @classmethod
    def _op(cls, op_name, *args, name=None):
        """Run operator (forward propagation).

        Parameters
        ----------
        op_name : str
            Operator's name.
        args : QQTensors
            Arguments of QQTensor.
        name : str, optional
            The name of result tensor, by default None.

        Returns
        -------
        QQTensor
            Result tensor.
        """

        assert all([isinstance(arg, QQTensor) for arg in args]), \
            "Arguments of operator should be an instance of `QQTensor` : {}".format([type(arg.v) for arg in args])

        if name is None:
            name = '{}<{}>'.format(op_name, ','.join([arg.name for arg in args]))

        return cls(auto_forward(op_name, *args), op_name, args, name=name)

    def _handle_gradient_internal(self):
        if self._gradient_internal['count'] >= 1:
            self._gradient = x.dot(
                np.ones((self._gradient.shape[-1], self._gradient.shape[-1])))
            self._gradient_internal['count'] = 0
                    
    @staticmethod
    def matmul(x1, x2):
        return QQTensor._op('matmul', x1, x2)

    # End of Operators

    # Properties

    @property
    def T(self):
        """Obtain transposed tensor.

        Returns
        -------
        array-like
            Transposed tensor.
        """
        
        return self._op("transpose", self)

    @property
    def v(self):
        """Tensor's value, alias of QQTensor._value.

        Returns
        -------
        numpy.ndarray
            Tensor's value.
        """

        return self._value

    @v.setter
    def v(self, x):
        """Setter of tensor's value.

        Parameters
        ----------
        x : numpy.ndarray
            Value.
        """

        assert isinstance(x, np.ndarray)
        self._value = x

    @property
    def grad(self):
        """Tensor's gradient, alias of QQTensor._gradient.

        Returns
        -------
        numpy.ndarray
            Tensor's gradient.
        """
        
        return self._gradient

    @grad.setter
    def grad(self, x):
        """Setter of tensor's gradient.

        Parameters
        ----------
        x : numpy.ndarray
            Gradient value.
        """

        assert isinstance(x, np.ndarray) or isinstance(x,)
        
        self._gradient = x

    @property
    def diff(self):
        """Whether tensor is differentiable, alias of QQTensor._differentiable.

        Returns
        -------
        bool
            A boolean which indicates tensor is differentiable.
        """

        return self._differentiable

    @property
    def ndim(self):
        """Tensor's number of dimention.

        Returns
        -------
        int
            Number of dimention.
        """

        return np.ndim(self._value)

    @property
    def name(self):
        """Tensor's name, alias of QQTensor._name.

        Returns
        -------
        str
            Tensor's name.
        """

        return self._name
    
    @property
    def op(self):
        """Tensor's operator name, alias of QQTensor._operator

        Returns
        -------
        str
            Tensor's operator name.
        """
        return self._operator

    @property
    def params(self):
        """Tensor's parameters list, alias of QQTensor._params_pool.

        Returns
        -------
        list
            Tensor's parameters list.
        """

        return self._params_pool

    @property
    def parents(self):
        """Tensor's parents list, alias of QQTensor._parent.

        Returns
        -------
        list
            Tensor's parents list.
        """
        return self._parent

    @property
    def shape(self):
        """Tensor's shape, alias of QQTensor._value.shape.

        Returns
        -------
        Tuple
            Tensor's shape.
        """

        return self._value.shape

    # End of Properties

    def backward(self, partial=None):
        """Perform backward propagation.

        Parameters
        ----------
        partial : QQTensor, optional
            Parameter that you want to obtain the gradient, 
            if not given, compute all gradients from this tensor, 
            by default None.

        Returns
        -------
        numpy.ndarray or None
            Computed gradient. If partial is not given, return None.
        """

        def handle_backward(param):
            # If self is not a scalar, it will compute the dot product with a one-lile output.
            # To sum all the value along batch axis.
            if not param.diff:
                return
                
            rt = self._backward(param)

            if rt.grad.ndim == 0:
                rt.grad = np.ones(param.shape) * self.shape[0]
            elif param.ndim == 1:
                rt.grad = np.dot(rt.grad.T, np.ones((self.shape[0], 1))).reshape(-1)
            #elif rt._gradient_internal['count'] < 2:
            #    rt.grad = rt.grad.dot(np.ones_like(self.v))
            else:
                rt.grad = rt.grad
            
            rt.reset_internal()

        if partial is None:
            grad_list = []
            for param in self._params_pool:
                handle_backward(param)
                grad_list.append(param.grad)
            return grad_list
        else:
            handle_backward(partial)
            return partial.grad

    def _backward(self, partial):
        """Internal recursive function for backwarding.

        Parameters
        ----------
        partial : QQTensor
            The tensor you want to obtain the partial derivatives.

        Returns
        -------
        numpy.ndarray or 0
            Computed gradient. If the tensor isn't differentiable, return 0.
        """

        if not self._differentiable:
            return self
        elif self._differentiable and self is partial:
            self._gradient = np.array(1., dtype=TENSORQQ_DTYPE)
            self._gradient_internal['internal'] = True
            self._gradient_internal['count'] += 1
            return self

        if self._operator == 'param':
            return self
        else:
            return auto_backward(self._operator, partial, self)

    def reset_internal(self):
        self._gradient_internal = {'count': 0, 'internal': False}

class QQTrainer(object):
    pass

class QQFullyConnection:
    """Fully-connection, also called affine transform.
        Formula: Y = X W.T + B

    Parameters
    ----------
    weight : QQTensor
        Weight of FC.
    bias : QQTensor
        Bias of FC.
    use_bias : bool
        Whether to use bias.
    """

    def __init__(self, weight : QQTensor, bias : QQTensor, use_bias : bool):
        self.weight = weight
        self.bias = bias
        self.use_bias = use_bias

    def __call__(self, x):
        if self.use_bias:
            return QQTensor.matmul(x, self.weight) + self.bias
        else:
            return QQTensor.matmul(x, self.weight)

if __name__ == "__main__":
    pass
