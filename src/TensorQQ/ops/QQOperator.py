import sys
import numpy as np
import inspect
import logging

ops_list = ['param', 'add', 'sub', 'mul', 'div', 'matmul', 'relu', 'sigmoid', 'transpose']
ops_pool = {}


def register():
    """Register all ops into the ops_pool.
    """

    def check_operator(op):
        return inspect.isclass(op) and issubclass(op, QQOperator) and op.__name__ != 'QQOperator'
    
    for name, cls in inspect.getmembers(
        sys.modules['TensorQQ.ops'], 
        predicate=check_operator):
        assert name.lower() in ops_list, \
            '{} is not in ops_list (__all__)'.format(name)
        ops_pool[name.lower()] = cls
    
    logging.debug("Registered operators: {}".format(ops_pool))

def auto_forward(op_name, *args):
    op_name = op_name.lower()
    if op_name in ops_pool:
        return ops_pool[op_name].forward(*args)
    else:
        raise AttributeError('Invalid operator: {}'.format(op_name))

def auto_backward(op_name, partial, self_tensor):
    op_name = op_name.lower()
    if op_name in ops_pool:
        return ops_pool[op_name].backward(partial, self_tensor)
    else:
        raise AttributeError('Invalid operator: {}'.format(op_name))


class QQOperator:
    @classmethod
    def forward(cls, *args):
        logging.debug("Forward OP: {}, Args: {}".format(
            cls.__name__, [(arg.name, arg.shape) for arg in args]))

    @classmethod
    def backward(cls, partial, self_tensor):
        logging.debug("------------------------------------------------")
        logging.debug("Partial Info: {}, Shape: {}".format(partial.name, partial.shape))
        logging.debug("Backward OP: {}, Parents: {}".format(
            cls.__name__, [(parent.name, parent.shape) for parent in self_tensor.parents]))


"""
class ReLU(QQOperator):
    @classmethod
    def forward(cls, *args):
        super().forward(*args)
        return np.maximum(np.zeros(args[0].shape), args[0].v)
    
    @classmethod
    def backward(cls, partial, *parents):
        super().backward(partial, *parents)
        step = np.copy(parents[0].v)
        step[step > 0] = 1
        step[step <= 0] = 0
        step = np.sum(step, axis=0, keepdims=True)
        #step = np.mean(step, axis=0, keepdims=True)
        p_grad = parents[0]._backward(partial)
        
        # If partial is a bias, the gradient will be flattened,
        # otherwise it will transpose step (relu's own derivatives) first, then calculte the gradient.
        return np.multiply(step, p_grad).flatten() if p_grad.ndim == 1 else np.multiply(step.T, p_grad)
"""

def unit_test():
    register()
    print(ops_pool)

if __name__ == "__main__":
    unit_test()
