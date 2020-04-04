# TODO: Check RMSProp optimizer
#

import TensorQQ
import numpy as np

w_1 = TensorQQ.QQTensor.QQParameter([[0.2358, 0.2746], 
                                     [0.2789, 0.3586],
                                     [0.1357, 0.0458]], name='w_1')
b_1 = TensorQQ.QQTensor.QQParameter([[0.1],
                                     [0.1],
                                     [0.1]], name='b_1')
w_2 = TensorQQ.QQTensor.QQParameter(np.array([[0.2468, 0.3579, 0.2455]]), name='w_2')

b_2 = TensorQQ.QQTensor.QQParameter([[0.2]], name='b_2')

x = TensorQQ.QQTensor.QQVariable(np.array([[0.1275], 
                                           [0.9864]]), name='x')
y = TensorQQ.QQTensor.QQVariable(np.array([[0.5]]), name='y')

initializer = TensorQQ.QQInitializer.QQNormal()

layer1 = TensorQQ.QQLayer.QQDense(3, weight=None, bias=None, initializer=initializer, name='layer_1')
layer2 = TensorQQ.QQLayer.QQDense(1, weight=None, bias=None, initializer=initializer, name='layer_2')
#layer3 = TensorQQ.QQLayer.QQDense(1, weight=None, bias=None, initializer=initializer, name='layer_output')
activation_relu1 = TensorQQ.QQActivation.QQReLU(name='relu_1')
#relu2 = TensorQQ.QQActivation.QQReLU(name='relu_2')
activation_sig = TensorQQ.QQActivation.QQSigmoid(name='sigmoid_1')
activation_tanh = TensorQQ.QQActivation.QQTanh(name= 'tanh_1')
loss_mse = TensorQQ.QQLoss.QQMSE(name='mse_1')

optimizer_sgd = TensorQQ.QQOptimizer.QQSGD(0.0001)
optimizer_rmsp = TensorQQ.QQOptimizer.QQRMSProp(0.01)
optimizer_adam = TensorQQ.QQOptimizer.QQAdam(0.0001)



for ep in range(5000):
    l1 = layer1(x)
    l1_a = activation_relu1(l1)
    out = layer2(l1_a)
    #l2_a = relu2(l2)
    #out = layer3(l2_a)
    loss = loss_mse(y, out)
    print('epoch', ep, 'output:', out.out, 'err:', loss.out)

    loss.backward()
    optimizer_sgd.update(loss, 1)

#print(layer1)
#print(layer2)
print('Done.')
