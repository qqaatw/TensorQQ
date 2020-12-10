import numpy as np
import torch
from torch.autograd import grad


# Forward
complex_A = np.arange(60).reshape((5, 2, 3, 2))

complex_B = np.arange(100).reshape((5, 2, 2, 5))

X1 = np.arange(15, dtype='float32').reshape((5, 3))

W1 = np.arange(18, dtype='float32').reshape((6, 3))

Y1 = np.matmul(X1, W1.T)

X2 = Y1

W2 = np.arange(24, dtype='float32').reshape((4, 6))

Y2 = np.matmul(X2, W2.T)

temp = []
for batch in range(X1.shape[0]):
    temp.append(X1[batch].reshape(-1, 1).dot(Y1[batch].reshape(1, -1)))

#input(np.array(temp, dtype='float32').sum(axis=0))

dW1 = X1.T.dot(np.ones_like(Y1)).dot(Y1)
#dW2 = np.dot(X2.T, Y2)


L1 = torch.nn.Linear(3, 6, False)
L1.weight = torch.nn.Parameter(torch.tensor(W1, requires_grad=True))
L2 = torch.nn.Linear(6, 4, False)
L2.weight = torch.nn.Parameter(torch.tensor(W2, requires_grad=True))

X_torch = torch.tensor(X1)
Y_torch = L1(X_torch)
Y_torch.backward(torch.ones_like(Y_torch))

print('Scratch')
print(X1.shape, W1.shape)
print(X2.shape, W2.shape)
print(Y1.shape, Y2.shape)
print(Y1)
print('dW1 dW2')
print(dW1.T)
#print(dW2.T)

print('Torch')
print(Y_torch)
print(L1.weight.grad)
print(L2.weight.grad)

print()



# Backward DY/DB

