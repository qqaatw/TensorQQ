# TODO: LSTM

from .QQTensor import QQBase
from .QQTensor import QQParameter
from .QQTensor import QQTensor
from .QQTensor import QQFullyConnection
from . import QQActivation
from . import QQUtility
import numpy as np


class QQLayer(QQBase):
    """Abstract layer class.

    Parameters
    ----------
    unit : int
        Output unit.
    name : str
        Custom Layer name.
    initializer : QQInitializer, optional
        Initializer, by default None.
    """
    
    def __init__(self, unit, name, initializer=None):
        super(QQLayer, self).__init__(name)
        self.unit = unit
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
    """Dense layer, also called fully-connected layer.

    Parameters
    ----------
    unit : int
        Output unit.
    weight : QQTensor, optional
        Initial weight tensor, by default None.
    bias : QQTensor, optional
        Initial bias tensor, by default None.
    weight_initializer : QQInitializer, optional
        Weight initializer, by default None.
    bias_initializer : QQInitializer, optional
        Bias initializer, 
        if bias_initializer is not given but weight_initializer is given, 
        uses weight_initializer instead, by default None.
    use_bias : bool, optional
        Whether to use bias.
    name : str, optional
        Custom layer name, by default 'undefined'.
    """

    def __init__(self, unit, weight=None, bias=None, weight_initializer=None, use_bias=True, bias_initializer=None, name='undefined'):
        super(QQDense, self).__init__(unit, name)
        self.weight['weight_1'] = weight
        self.bias['bias_1'] = bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = weight_initializer if not bias_initializer and weight_initializer else bias_initializer
        self.use_bias = use_bias
        self.fc = None

    def __call__(self, x):
        super().__call__(x)
        self.out = self.fc(x)
        return self.out

    def init(self, x):
        self._initialized = True
        if self.weight['weight_1'] is None:
            if self.weight_initializer:
                self.weight['weight_1'] = QQTensor(
                    self.weight_initializer((self.unit, x.shape[1])),
                    name='{}_{}'.format(self.name, 'weight_1')
                )
            else:
                raise RuntimeError('Weight initializer is not given!')

        if self.use_bias and self.bias['bias_1'] is None:
            if self.bias_initializer:
                self.bias['bias_1'] = QQTensor(
                    self.bias_initializer((self.unit,)),
                    name='{}_{}'.format(self.name, 'bias_1')
                )
            else:
                self.bias['bias_1'] = QQTensor(
                    self.weight_initializer((self.unit,)),
                    name='{}_{}'.format(self.name, 'bias_1')
                )

        #elif self.weight['weight_1'] is None or self.bias['bias_1'] is None:
        #    raise ValueError(
        #        '{}\'s weight or bias has not been initialized!'.format(self.name))
        assert isinstance(self.weight['weight_1'], QQTensor), \
            '{}\'s weight is not a QQTensor instance.'.format(self.name)
        assert self.weight['weight_1'].shape == (x.shape[1], self.unit),\
            '{}\'s weight has an invalid shape: {}.'.format(self.name, self.weight['weight_1'].shape)
        if self.use_bias:
            assert isinstance(self.bias['bias_1'], QQTensor), \
                '{}\'s bias is not a QQTensor instance.'.format(self.name)
            assert self.bias['bias_1'].shape == (self.unit,),\
                '{}\'s bias has an invalid shape: {}.'.format(self.name, self.bias['bias_1'].shape)
            
        self.fc = QQFullyConnection(self.weight['weight_1'], self.bias['bias_1'], use_bias=self.use_bias)


class QQDense_deprecated(QQLayer):
    """Dense layer, also called fully-connected layer.

    Parameters
    ----------
    unit : int
        Output unit.
    weight : QQParameter, optional
        Initial weight tensor, by default None
    bias : QQParameter, optional
        Initial bias tensor, by default None
    initializer : QQInitializer, optional
        Initial initializer, by default None
    name : str, optional
        Tensor name, by default 'undefined'
    """

    def __init__(self, unit, weight=None, bias=None, initializer=None, name='undefined'):
        super().__init__(unit, name, initializer)
        self.weight['weight_1'] = weight
        self.bias['bias_1'] = bias

    def __call__(self, x):
        super().__call__(x)
        #self.out = np.matmul(
        #    self.weight['weight_1'].weight, x.out) + self.bias['bias_1'].weight
        self.out = QQUtility.fc(x=x.out, weight=self.weight['weight_1'].weight, bias=self.bias['bias_1'].weight)
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


class QQLSTM(QQLayer):
    """Long short term memory (LSTM) layer.

    Parameters
    ----------
    unit : int
        Output unit.
    weight : Tuple of QQParameters, optional
        Initial weight tensors, 
        shape: (f_gate_i2h, i_gate_i2h, o_gate_i2h, g_gate_i2h,
                f_gate_h2h, i_gate_h2h, o_gate_h2h, g_gate_h2h),
        by default None
    bias : Tuple of QQParameters, optional
        Initial bias tensors, shape: (f_gate, i_gate, o_gate, g_gate), by default None
    initializer : QQInitializer, optional
        Initial initializer, by default None
    name : str, optional
        Tensor name, by default 'undefined'

    Inputs
    ------
    x : QQVariable
        Input tensor, shape: (timesteps, channels).
    """
    
    def __init__(self, unit, weight=(None,) * 8, bias=(None,) * 4, initializer=None, name='undefined'):
        super(QQLSTM, self).__init__(unit, name, initializer)
        assert len(weight) == 8, \
            "Length of weight should be 8: (8 vs {})".format(len(weight))
        assert len(bias) == 4, \
            "Length of bias should be 4: (4 vs {})".format(len(bias))
        
        (self.weight['f_gate_i2h'], self.weight['i_gate_i2h'],
        self.weight['o_gate_i2h'], self.weight['g_gate_i2h'],
        self.weight['f_gate_h2h'], self.weight['i_gate_h2h'], 
        self.weight['o_gate_h2h'], self.weight['g_gate_h2h']) = weight

        self.bias['f_gate'], self.bias['i_gate'], self.bias['o_gate'], self.bias['g_gate'] = bias
    
        self.states = []

    def __call__(self, x):
        super(QQLSTM, self).__call__(x)
        state = self.set_begin_state()
        timesteps, channels = self.x['x'].shape
        sigmoid = QQActivation.QQSigmoid()
        tanh = QQActivation.QQTanh()

        for step in range(timesteps + 1):
            input_i = np.expand_dims(self.x['x'].out[step], 2)

            f_gate = sigmoid(np.matmul(self.weight['f_gate_i2h'], input_i) + \
                     np.matmul(self.weight['f_gate_h2h'], state[1]) + self.bias['f_gate'])
            i_gate = sigmoid(np.matmul(self.weight['i_gate_i2h'], input_i) + \
                     np.matmul(self.weight['i_gate_h2h'], state[1]) + self.bias['i_gate'])
            o_gate = sigmoid(np.matmul(self.weight['o_gate_i2h'], input_i) + \
                     np.matmul(self.weight['o_gate_h2h'], state[1]) + self.bias['o_gate'])
            g_gate = tanh(np.matmul(self.weight['g_gate_i2h'], input_i) + \
                     np.matmul(self.weight['g_gate_h2h'], state[1]) + self.bias['g_gate'])
            
            c = np.matmul(f_gate, state[0]) + np.matmul(i_gate, g_gate)
            h = np.matmul(o_gate, tanh(c))
            state = (c, h)

            self.states.append(state)
        
        return state[1], self.states

    def backward(self):
        steps = len(self.states)
        for step in range(steps - 1, -1, -1):
            pass
    
    def init(self, x):
        if self.initializer:
            for i, (key, value) in enumerate(self.weight.items()):
                in_shape = x.shape[0]
                if value is None:
                    self.weight[key] = QQParameter(
                    initializer=self.initializer, name="{}_{}".format(self.name, key))
                    self.weight[key].init((self.unit, in_shape), is_bias=False)
                else:
                    assert self.weight[key].shape == (self.unit, in_shape),\
                        '{}\'s weight has an invalid shape. ({} vs {})'.format(
                            self.name,  (self.unit, in_shape),  self.weight[key].shape)

            for i, (key, value) in enumerate(self.bias.items()):
                if value is None:
                    self.bias[key] = QQParameter(
                    initializer=self.initializer, name="{}_{}".format(self.name, key))
                    self.bias[key].init((self.unit, 1), is_bias=True)
                else:
                    assert self.bias[key].shape == (self.unit, 1),\
                        '{}\'s weight has an invalid shape. ({} vs {})'.format(
                            self.name,  (self.unit, 1),  self.bias[key].shape)

        elif not all(self.weight.values()) or not all(self.bias.values()):
            raise ValueError('{}\'s weight or bias has not been initialized!'.format(self.name))

        # Collect parameters
        self.params_pool += list(self.weight.values())
        self.params_pool += list(self.bias.values())
        self.params_pool = self.params_pool + x.params_pool
        
        self._initialized = True

    def set_begin_state(self):
        state = (
        np.zeros((self.unit, 1)),
        np.zeros((self.unit, 1))
        )
        self.states.append(state)
        return state
