# 0x04. Convolutions and Pooling

## Description
What you should learn from this project:

* What is a convolution?
* What is max pooling? average pooling?
* What is a kernel/filter?
* What is padding?
* What is “same” padding? “valid” padding?
* What is a stride?
* What are channels?
* How to perform a convolution over an image
* How to perform max/average pooling over an image

---

### [0. Valid Convolution](./0-convolve_grayscale_valid.py)
* Write a function def convolve_grayscale_valid(images, kernel): that performs a valid convolution on grayscale images:


### [1. Same Convolution](./1-convolve_grayscale_same.py)
* Write a function def convolve_grayscale_same(images, kernel): that performs a same convolution on grayscale images:


### [2. Convolution with Padding](./2-convolve_grayscale_padding.py)
* Write a function def convolve_grayscale_padding(images, kernel, padding): that performs a convolution on grayscale images with custom padding:


### [3. Strided Convolution](./3-convolve_grayscale.py)
* Write a function def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on grayscale images:


### [4. Convolution with Channels](./4-convolve_channels.py)
* Write a function def convolve_channels(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on images with channels:


### [5. Multiple Kernels](./5-convolve.py)
* Write a function def convolve(images, kernels, padding='same', stride=(1, 1)): that performs a convolution on images using multiple kernels:


### [6. Pooling](./6-pool.py)
* Write a function def pool(images, kernel_shape, stride, mode='max'): that performs pooling on images:

---

## Author
* **Nicolas Martinez Machado** - [Noeuclides](https://github.com/Noeuclides)