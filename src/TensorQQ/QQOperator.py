from TensorQQ import QQTensor

import sys
import numpy as np 
import inspect

ops_list = ['param', 'add', 'sub', 'mul', 'div', 'matmul', 'relu']
ops_pool = {}


def register():
    def check_operator(op):
        return inspect.isclass(op) and issubclass(op, QQOperator) and op.__name__ != 'QQOperator'
    
    for name, cls in inspect.getmembers(
        sys.modules['TensorQQ.QQOperator'], 
        predicate=check_operator):
        assert name.lower() in ops_list, \
            '{} is not in ops_list'.format(name)
        ops_pool[name.lower()] = cls


def shared_mul_div(partial, parent_0, parent_1):
    grad_p0 = None
    grad_p1 = None

    # Gradients don't exist in grad_p0 or grad_p1.
    if partial not in parent_0.params and \
        partial not in parent_1.params:
        return 0, 0


    if partial in parent_0.params:
        grad_p0 = parent_0._backward(partial)
    else:
        grad_p0 = np.sum(parent_0.v, axis=0)

    if partial in parent_1.params:
        grad_p1 = parent_1._backward(partial)
    else:
        grad_p1 = np.sum(parent_1.v, axis=0)

    #print("parent0", parent_0.name, '\n', parent_0.shape)
    #print("parent1", parent_1.name, '\n', parent_1.shape)
       
    #grad_p0 = parent_0.v if grad_p0 is None else grad_p0
    #grad_p1 = parent_1.v if grad_p1 is None else grad_p1
    print('grads:', grad_p0, grad_p1)
    #grad_p0 = np.sum(grad_p0, axis=0)
    #grad_p1 = np.sum(grad_p1, axis=0)
    print('sum_grads', grad_p0, grad_p1)
    return grad_p0, grad_p1

def sqrt(x):
    y = np.sqrt(x)
    y[np.isnan(y)] = 0.
    return y

def fc(x, weight, bias):
    return np.matmul(weight, x) + bias

def auto_forward(op_name, *args):
    op_name = op_name.lower()
    if op_name in ops_pool:
        return ops_pool[op_name].forward(*args)
    else:
        raise AttributeError('Invalid operator: {}'.format(op_name))


def auto_backward(op_name, partial, *parents):
    op_name = op_name.lower()
    if op_name in ops_pool:
        return ops_pool[op_name].backward(partial, *parents)
    else:
        raise AttributeError('Invalid operator: {}'.format(op_name))

class QQOperator:
    @classmethod
    def forward(cls, *args):
        raise NotImplementedError

    @classmethod
    def backward(cls, partial, *parents):
        raise NotImplementedError

class Matmul(QQOperator):
    @classmethod
    def forward(cls, *args):
        return np.matmul(args[0].v, args[1].v)
    
    @classmethod
    def backward(cls, partial, *parents):
        grad_p0, grad_p1 = shared_mul_div(partial, parents[0], parents[1])
        return np.multiply(grad_p0, grad_p1)

class ReLU(QQOperator):
    @classmethod
    def forward(cls, *args):
        return np.maximum(np.zeros(args[0].shape), args[0].v)
    
    @classmethod
    def backward(cls, partial, *parents):
        step = np.copy(parents[0].v)
        step[step > 0] = 1
        step[step <= 0] = 0
        return np.multiply(step, parents[0]._backward(partial))

def unit_test():
    register()
    print(ops_pool)

if __name__ == "__main__":
    unit_test()
