#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('3-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('0-one_hot_encode').one_hot_encode
one_hot_decode = __import__('1-one_hot_decode').one_hot_decode

lib= np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)

deep = Deep.load('3-saved.pkl')
A_one_hot, cost = deep.train(X_train, Y_train_one_hot, iterations=100,
                             step=10, graph=False)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_train == A) / Y_train.shape[0] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

A_one_hot, cost = deep.evaluate(X_valid, Y_valid_one_hot)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_valid == A) / Y_valid.shape[0] * 100
print("Validation cost:", cost)
print("Validation accuracy: {}%".format(accuracy))

deep.save('3-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_valid_3D[i])
    plt.title(A[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
ubuntu@alexa-ml:~$ ./3-main.py
Cost after 0 iterations: 0.44256925735606084
Cost after 10 iterations: 0.44145627418970107
Cost after 20 iterations: 0.4403564343612417
Cost after 30 iterations: 0.43926952193335933
Cost after 40 iterations: 0.4381953250503073
Cost after 50 iterations: 0.4371336358442412
Cost after 60 iterations: 0.4360842503447979
Cost after 70 iterations: 0.43504696839171103
Cost after 80 iterations: 0.4340215935503086
Cost after 90 iterations: 0.433007933029748
Cost after 100 iterations: 0.4320057976038559
Train cost: 0.4320057976038559
Train accuracy: 88.36%
Validation cost: 0.3986256380175584
Validation accuracy: 89.67%
