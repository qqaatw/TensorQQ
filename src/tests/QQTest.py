# TODO: Check RMSProp optimizer
#

import numpy as np

from TensorQQ import *

# Create input tensors and parameters
init_normal = QQInitializer.QQNormal()
init_uniform = QQInitializer.QQUniform()

w_1 = QQTensor.QQParameter([[0.2358, 0.2746], 
                            [0.2789, 0.3586],
                            [0.1357, 0.0458]], initializer=init_normal, name='w_1')
b_1 = QQTensor.QQParameter([[0.1],
                            [0.1],
                            [0.1]], name='b_1')
w_2 = QQTensor.QQParameter(np.array([[0.2468, 0.3579, 0.2455]]), name='w_2')

b_2 = QQTensor.QQParameter([[0.2]], name='b_2')

x = QQTensor.QQVariable(np.array([[0.1275], 
                                  [0.9864]]), name='x')
y = QQTensor.QQVariable(np.array([[0.5]]), name='y')


lstm_x = QQTensor.QQVariable(np.random.normal(size=(3, 5)))


# Create layers, activations and loss
layer1 = QQLayer.QQDense_deprecated(3, weight=None, bias=None, initializer=init_normal, name='layer_1')
layer2 = QQLayer.QQDense_deprecated(1, weight=None, bias=None, initializer=init_uniform, name='layer_2')
layer3 = QQLayer.QQDense_deprecated(1, weight=None, bias=None, initializer=init_uniform, name='layer_output')
layer_lstm_1 = QQLayer.QQLSTM(10, initializer=init_normal, name='lstm_1')
act_relu1 = QQActivation.QQReLU_deprecated(name='relu_1')
act_relu2 = QQActivation.QQReLU_deprecated(name='relu_2')
act_sig = QQActivation.QQSigmoid_deprecated(name='sigmoid_1')
act_tanh1 = QQActivation.QQTanh_deprecated(name= 'tanh_1')
act_tanh2 = QQActivation.QQTanh_deprecated(name= 'tanh_2')
loss_mse = QQLoss.QQMSE(name='mse_1')

#a, b = layer_lstm_1(lstm_x)


# Create optimizer
optimizer_sgd = QQOptimizer.QQSGD(0.0001)
optimizer_rmsp = QQOptimizer.QQRMSProp(0.01)
optimizer_adam = QQOptimizer.QQAdam(0.0001)


for ep in range(5000):
    l1 = layer1(x)
    l1_a = act_tanh1(l1)
    l2 = layer2(l1_a)
    l2_a = act_tanh2(l2)
    out = layer3(l2_a)
    loss = loss_mse(y, out)
    print('epoch', ep, 'output:', out.out, 'err:', loss.out)

    loss.backward()
    optimizer_adam.update(loss, 1)

print(layer1)
print(layer2)
print('Done.')
