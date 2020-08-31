import numpy as np 
import sys
import matplotlib
import nnfs
from nnfs.datasets import spiral_data
import math

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

class Activation_Softmax:
    def forward(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred) #number of samples
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) #clip data for at undgå divison af 0
        y_pred_clipped = y_pred_clipped[range(samples), y_true] #sandsynlighed for target value
        negative_log_likelihoods = -np.log(y_pred_clipped) #losses
        return negative_log_likelihoods

dense1 = Layer_Dense(2,3)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print(f"Loss: {loss}")

predictions = np.argmax(activation2.output, axis=1)
accuracy = np.mean(predictions==y)

print(f"Accuracy: {accuracy}")
