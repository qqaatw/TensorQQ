# TODO:
# Solve multipled gradients problems.
# QQLose
# QQOptimizer

import argparse
import logging

from TensorQQ import QQTensor, QQInitializer, QQActivation, QQLayer, QQOptimizer
 
import numpy as np

MX_DEBUG = None
logger = logging.getLogger()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', help='Test options',
        type=str, default='general', choices=['general', 'simple'])
    parser.add_argument('--debug', help='Debug mode.',
        action='store_true')
    parser.add_argument('--mxnet', help='Toggle on MxNET debugging code.',
        action='store_true')
    parser.add_argument('--tensorflow', help='Toggle on TensorFlow debugging code.',
        action='store_true')
    return parser

# First layer's weight
"""
l1_weight_array = [[-0.97352848, -0.38923787],
                [0.7398363,  0.25032709],
                [-1.15975133, -0.04092976],
                [0.03388288, -1.17805154],
                [0.87419297, -0.68605638]]
"""
l1_weight_array = [[-0.97352848,  0.7398363, -1.15975133,  0.03388288,  0.87419297],
                   [-0.38923787,  0.25032709, -0.04092976, -1.17805154, -0.68605638]]

l1_weight = QQTensor.QQTensor(l1_weight_array, name='layer_1_weight_1')

# Second layer's weight
"""
l2_weight_array = [[-1.66390359, -0.73179574,  0.36323441,  0.85232713, -1.27643517],
                   [0.11640474, -0.60168938,  0.22826849,  1.52825402,  0.65185428],
                   [0.72830206,  0.88843326, -0.68269403,  0.27281709, -0.9148804]]
"""
l2_weight_array = [[-1.66390359,  0.11640474,  0.72830206],
       [-0.73179574, -0.60168938,  0.88843326],
       [0.36323441,  0.22826849, -0.68269403],
       [0.85232713,  1.52825402,  0.27281709],
       [-1.27643517,  0.65185428, -0.9148804]]

l2_weight = QQTensor.QQTensor(l2_weight_array, name='layer_2_weight_1')

# First layer's bias
l1_bias_batch_array = [-0.97352848, -0.38923787, 0.7398363, 0.25032709, -1.15975133]
l1_bias_batch = QQTensor.QQTensor(l1_bias_batch_array, name='layer_1_bias_1')

# Second layer's bias
l2_bias_batch_array = [-0.30378208, 0.72830206, 
0.88843326]
l2_bias_batch = QQTensor.QQTensor(l2_bias_batch_array, name='layer_2_bias_1')

# Input
#x_batch_array = [[0.5, 0.5],
#                 [1.5, 1.5]]
x_batch_array = [[0.5, 1.7],
                 [2.5, 3.3],
                 [4.5, 5.6]]
X_batch = QQTensor.QQTensor(x_batch_array, differentiable=False, name='x')

def mx_debug(x, layer_params_list):
    # TODO: 
    print("MXNET Debug Result:")
    import mxnet as mx
    mx_weight = mx.nd.array(l1_weight_array)
    mx_weight.attach_grad()
    mx_batch_bias = mx.nd.array(l1_bias_batch_array)
    mx_batch_bias.attach_grad()
    mx_batch_x = mx.nd.array(x_batch_array)

    #mx_out = mx.sym.FullyConnected(data=mx_X, weight=mx_weight, bias=mx_bias)
    print(mx_batch_x.shape, mx_weight.shape, mx_batch_bias.shape)
    with mx.autograd.record():
        mx_out = mx.nd.add(mx.nd.dot(mx_batch_x, mx_weight.T), mx_batch_bias)
        #mx_out = mx.nd.relu(mx_out)
        mx_grad = mx.autograd.grad(mx_out, [mx_weight, mx_batch_bias])
    print('mx_out', mx_out)
    print('mx_grad', mx_grad)
    """
    mx_dense = mx.gluon.nn.Dense(10)
    mx_dense.initialize()
    mx_dense_out = mx_dense(mx_batch_x)
    print(mx_dense.collect_params())
    print('mx_dense_out', mx_dense_out)
    """

def tf_debug(x, layer_params_list):
    import tensorflow as tf
    tf.executing_eagerly()
    tf_batch_x = tf.constant(x_batch_array)
    l1_tf_batch_weight = tf.Variable(l1_weight_array)
    l1_tf_batch_bias = tf.Variable(l1_bias_batch_array)
    l2_tf_batch_weight = tf.Variable(l2_weight_array)
    l2_tf_batch_bias = tf.Variable(l2_bias_batch_array)
    with tf.GradientTape(persistent=True) as tape:
        tf_out = tf.matmul(
            tf_batch_x,
            l1_tf_batch_weight) + l1_tf_batch_bias

        tf_out = tf.nn.relu(tf_out)
        
        #tf_out = tf.matmul(
        #    tf_out,
        #    l2_tf_batch_weight) + l2_tf_batch_bias
        #tf_out = tf.nn.relu(tf_out)
    tf_grad = tape.gradient(tf_out, [l1_tf_batch_weight, l1_tf_batch_bias, l2_tf_batch_weight, l2_tf_batch_bias])
    print("tf_out", tf_out)
    print("tf_grad", tf_grad)
    del tape


def simple_one_layer_test():
    fc = QQTensor.QQFullyConnection(weight, l1_bias_batch)
    # (5*2) * (2*1) + (5*1) = 5 * 1
    out = fc(X_batch)
    grad = out.backward(fc.weight)
    print('x', X_batch.v, X_batch.shape)
    print('weight', weight.v)
    print('bias', l1_bias_batch.v)
    print('out', out.v)
    print('grad', grad, grad.shape)

def general_test():
    init_zeros = QQInitializer.QQZero()
    init_normal = QQInitializer.QQNormal()
    init_uniform = QQInitializer.QQUniform()
    #optimizer_sgd = QQOptimizer.QQSGD()
    layer1 = QQLayer.QQDense(5, weight=l1_weight, bias=l1_bias_batch,
                            weight_initializer=None, bias_initializer=init_zeros, name='layer_1')
    act1 = QQActivation.QQReLU()
    act2 = QQActivation.QQReLU()
    layer2 = QQLayer.QQDense(3, weight=l2_weight, bias=l2_bias_batch,
                            weight_initializer=init_uniform, name='layer_2')
    layer3 = QQLayer.QQDense(1, weight=None, bias=None,
                            weight_initializer=init_uniform, name='layer_output')


    out = layer1(X_batch)
    out = act1(out)
    #out = layer2(out)
    #out = act2(out)

    final_out = out
    final_out.backward()
    print('OK', f"{final_out.v}\n", 
        *[ f"{i.name}, {i.shape}, {i.grad.shape}\n{i.grad}\n" for i in final_out.params if i.diff])

if __name__ == "__main__":
    args = get_parser().parse_args()
    
    if args.debug:
        logger.setLevel('DEBUG')
    else:
        logger.setLevel('INFO')
    if args.mxnet:
        #MX_DEBUG = True
        mx_debug(None, None)
    else:
        #MX_DEBUG = False
        pass

    if args.tensorflow:
        tf_debug(None, None)

    if args.option == "simple":
        simple_one_layer_test()
    elif args.option == "general":
        general_test()
