# 0x06. Keras

## Description
What you should learn from this project:

* What is Keras?
* What is a model?
* How to instantiate a model (2 ways)
* How to build a layer
* How to add regularization to a layer
* How to add dropout to a layer
* How to add batch normalization
* How to compile a model
* How to optimize a model
* How to fit a model
* How to use validation data
* How to perform early stopping
* How to measure accuracy
* How to evaluate a model
* How to make a prediction with a model
* How to access the weights/outputs of a model
* What is HDF5?
* How to save and load a model’s weights, a model’s configuration, and the entire model

---

### [0. Sequential](./0-sequential.py)
* Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library:


### [1. Input](./1-input.py)
* Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library:


### [2. Optimize](./2-optimize.py)
* Write a function def optimize_model(network, alpha, beta1, beta2): that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics:


### [3. One Hot](./3-one_hot.py)
* Write a function def one_hot(labels, classes=None): that converts a label vector into a one-hot matrix:


### [4. Train](./4-train.py)
* Write a function def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False): that trains a model using mini-batch gradient descent:


### [5. Validate](./5-train.py)
* Based on 4-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False): to also analyze validaiton data:


### [6. Early Stopping](./6-train.py)
* Based on 5-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False): to also train the model using early stopping:


### [7. Learning Rate Decay](./7-train.py)
* Based on 6-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False): to also train the model with learning rate decay:


### [8. Save Only the Best](./8-train.py)
* Based on 7-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False): to also save the best iteration of the model:


### [9. Save and Load Model](./9-model.py)
* Write the following functions:


### [10. Save and Load Weights](./10-weights.py)
* Write the following functions:


### [11. Save and Load Configuration](./11-config.py)
* Write the following functions:


### [12. Test](./12-test.py)
* Write a function def test_model(network, data, labels, verbose=True): that tests a neural network:


### [13. Predict](./13-predict.py)
* Write a function def predict(network, data, verbose=False): that makes a prediction using a neural network:

---

## Author
* **Nicolas Martinez Machado** - [Noeuclides](https://github.com/Noeuclides)