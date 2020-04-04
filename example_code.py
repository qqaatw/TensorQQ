import numpy as np

class Layer(object):
    def __init__(self, weight=None, bias=None, name = 'undefined'):
        self.name = name
        self.weight = weight
        self.bias = bias

        self.grad_weight = None
        self.grad_bias = None

    def __call__(self, x):
        return np.matmul(self.weight, x) + self.bias

    def backward(self, x):
        self.grad_weight = x
        self.grad_bias = 1

def MSE(t, p):
    return (t-p) * (t-p)

def ReLU(x):
    return np.maximum(np.zeros(x.shape), x)

def d_ReLU(x):
    y = x.copy()
    y[y > 0] = 1
    y[y<= 0] = 0
    return y

def show_parameters(*v, x, y):
    for _ in v:
        print(_.name, ' =\nweight', _.weight, _.weight.shape, '\nbias', _.bias, _.bias.shape)
    print('x =\n', x)
    print('y =\n', y)

w_1 = np.array(
    [[0.2358, 0.2746], 
     [0.2789, 0.3586], 
     [0.1357, 0.0458]]
)

b_1 = np.array(
    [[0.1],
     [0.1],
     [0.1]]
)

w_2 = np.array([[0.2468, 0.3579, 0.2455]])

b_2 = np.array([[0.2]])

x = np.array([[0.1275],
              [0.9864]])
t = np.array([[0.5]])

layer_1 = Layer(w_1, b_1, 'layer1')
layer_2 = Layer(w_2, b_2, 'layer2')

learning_rate = 0.01

show_parameters(layer_1, layer_2, x = x, y = t)


for ep in range(1000):
    # feed-forward
    l1 = layer_1(x)
    l1_a = ReLU(l1)
    l2 = layer_2(l1_a)
    err = MSE(t, l2)
    print('epoch', ep, 'output:', l2, 'err:', err)

    # backward (d_w's shape may differ to w)
    d_w_1 = 2 * (l2 - t) * np.dot(np.dot(layer_2.weight, d_ReLU(l1)),  x.T)
    d_w_2 = 2 * (l2 - t) * l1_a.T 

    d_b_1 = 2 * (l2 - t) * np.dot(layer_2.weight, d_ReLU(l1))
    d_b_2 = 2 * (l2 - t)

    # GD network update
    layer_1.weight = layer_1.weight - learning_rate * d_w_1
    layer_2.weight = layer_2.weight - learning_rate * d_w_2
    layer_1.bias = layer_1.bias - learning_rate * d_b_1
    layer_2.bias = layer_2.bias - learning_rate * d_b_2

show_parameters(layer_1, layer_2, x=x, y=t)


