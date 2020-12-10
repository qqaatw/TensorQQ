import numpy as np

from .QQOperator import QQOperator

__all__ = ['Add', 'Sub', 'Mul', 'Div', 'Matmul', 'Transpose']


def shared_mul_div(partial, st):
    """Backward matmul.

    Parameters
    ----------
    partial : QQTensor
        The tensor you want to obtain the partial derivatives.
    st : QQTensor
        Self tensor.

    Returns
    -------
    QQTensor
        Result tensor.
    """

    exist_0, exist_1 = (partial in st.parents[0].params,
                        partial in st.parents[1].params)

    # Gradients exist
    if any((exist_0, exist_1)):
        if all((exist_0, exist_1)):
            raise AssertionError('The circumstance of parameters sharing. Currently under development, check first.')
            return st.parents[0].v.T * st.parents[1].v.T
        
        
        if exist_1:
            # A@B'
            rt = st.parents[1]._backward(partial) # rt = partial 
            A = st.parents[0].v
            B = rt.grad
            rt.grad = np.dot(A.T, B) # x @ W=1
        else:
            # A'@B
            rt = st.parents[0]._backward(partial)
            A = rt.grad
            B = st.parents[1].v
            rt.grad = np.dot(A, B.T)
        
        return rt

    # Gradients don't exist in st.parents[0] or st.parents[1].
    return None


class Add(QQOperator):
    @classmethod
    def forward(cls, *args):
        super().forward(*args)
        return args[0].v + args[1].v

    @classmethod
    def backward(cls, partial, self_tensor):
        super().backward(partial, self_tensor)
        
        rt1 = self_tensor.parents[0]._backward(partial)
        rt2 = self_tensor.parents[1]._backward(partial)

        # check if gradient exists
        A = rt1.grad if rt1 and rt1._gradient_internal['internal'] else 0
        B = rt2.grad if rt2 and rt2._gradient_internal['internal'] else 0
        partial.grad = np.array(A + B)

        return partial


class Sub(QQOperator):
    @classmethod
    def forward(cls, *args):
        super().forward(*args)
        return args[0].v - args[1].v

    @classmethod
    def backward(cls, partial, self_tensor):
        super().backward(partial, self_tensor)
        return self_tensor.parents[0]._backward(partial) - self_tensor.parents[1]._backward(partial)


class Mul(QQOperator):
    @classmethod
    def forward(cls, *args):
        super().forward(*args)
        return args[0].v * args[1].v

    @classmethod
    def backward(cls, partial, self_tensor):
        super().backward(partial, self_tensor)
        grad_p0, grad_p1 = shared_mul_div(
            partial, self_tensor.parents[0], self_tensor.parents[1])
        return grad_p0 * grad_p1


class Div(QQOperator):
    @classmethod
    def forward(cls, *args):
        super().forward(*args)
        return args[0].v / args[1].v

    @classmethod
    def backward(cls, partial, self_tensor):
        super().backward(partial, self_tensor)
        grad_p0, grad_p1 = shared_mul_div(
            partial, self_tensor.parents[0], self_tensor.parents[1])
        return grad_p0 / grad_p1


class Matmul(QQOperator):
    @classmethod
    def forward(cls, *args):
        super().forward(*args)
        return np.matmul(args[0].v, args[1].v)

    @classmethod
    def backward(cls, partial, self_tensor):
        super().backward(partial, self_tensor)
        grad = shared_mul_div(partial, self_tensor)
        return grad


class Transpose(QQOperator):
    """https://math.stackexchange.com/questions/704773/what-is-the-derivative-of-a-vector-with-respect-to-its-transpose
    """

    @classmethod
    def forward(cls, *args):
        super().forward(*args)
        return np.transpose(args[0].v)

    @classmethod
    def backward(cls, partial, self_tensor):
        super().backward(partial, self_tensor)
        #return self_tensor.parents[0]._backward(partial)
        return np.transpose(self_tensor.parents[0]._backward(partial))
