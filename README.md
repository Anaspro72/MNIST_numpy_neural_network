# MNIST_numpy_neural_network
 # Neural Network from Scratch using Numpy
This repository contains a simple neural network model built from scratch using Numpy, without any external machine learning libraries like TensorFlow, PyTorch, or JAX. The model is designed to gain deep insights into the internal workings of neural networks. It is a fully functional neural network with three layers, implemented and tested on the popular MNIST dataset.
# Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)

# Overview
This neural network model is a simple implementation to better understand the mechanics of forward and backward propagation, weight updates, and activation functions. It demonstrates how neural networks can be built from the ground up using Numpy and provides a deeper understanding of the training process, loss functions, and optimization techniques.

The model is applied to the MNIST dataset, a set of 28x28 pixel grayscale images of handwritten digits. The goal is to classify these digits into one of the 10 possible classes (0-9).

# Features
Implemented using only Numpy (no external ML frameworks).\n
Three-layer architecture (input, hidden, output).\n
Utilizes ReLU and Softmax activation functions.\n
Gradient Descent optimization for weight updates.\n
Supports training and evaluation on the MNIST dataset.\n

# Getting Started
To get started with this project, you’ll need to clone this repository and install the necessary dependencies.

# Model Architecture
The neural network consists of three layers:

## Input Layer: 
The input layer consists of 784 nodes (one for each pixel of the 28x28 images).
## Hidden Layer:
A fully connected layer with ReLU activation function.
## Output Layer: 
A fully connected layer with Softmax activation for classification into 10 possible categories (digits 0-9).

# Training and Evaluation
## Training
The model is trained using Gradient Descent to minimize the loss function. Training proceeds in the following steps:

## Initialize weights and biases.
Perform forward propagation to compute predictions.
Compute the loss and backpropagate the error to update weights.

## Evaluation
The model's performance is evaluated on the MNIST test set after training. The accuracy of the model on unseen data is computed.

# Results
Accuracy: Achieved [90.79] accuracy on the MNIST dataset.
The model demonstrates the ability to classify handwritten digits
# Contributing
If you would like to contribute to this project, feel free to fork the repository and create a pull request with improvements. Contributions, issues, and feature requests are welcome.



