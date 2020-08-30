import numpy as np 
import sys
import matplotlib
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(activation1.output)






"""
inputs =  [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

weights =   [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases =    [2, 3, 0.5]

weights2 =   [[0.1, -1.14, -0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 =    [-1, 2, -0.4]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

"""