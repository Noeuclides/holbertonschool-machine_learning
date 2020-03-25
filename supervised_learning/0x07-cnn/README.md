# 0x07. Convolutional Neural Networks

## Description
What you should learn from this project:

* What is a convolutional layer?
* What is a pooling layer?
* Forward propagation over convolutional and pooling layers
* Back propagation over convolutional and pooling layers
* How to build a CNN using Tensorflow and Keras

---

### [0. Convolutional Forward Prop](./0-conv_forward.py)
* Write a function def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)): that performs forward propagation over a convolutional layer of a neural network:


### [1. Pooling Forward Prop](./1-pool_forward.py)
* Write a function def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs forward propagation over a pooling layer of a neural network:


### [2. Convolutional Back Prop](./2-conv_backward.py)
* Write a function def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)): that performs back propagation over a convolutional layer of a neural network:


### [3. Pooling Back Prop](./3-pool_backward.py)
* Write a function def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs back propagation over a pooling layer of a neural network:


### [4. LeNet-5 (Tensorflow)](./4-lenet5.py)
* 


### [5. LeNet-5 (Keras)](./5-lenet5.py)
* Write a function def lenet5(X): that builds a modified version of the LeNet-5 architecture using keras:


---

## Author
* **Nicolas Martinez Machado** - [Noeuclides](https://github.com/Noeuclides)