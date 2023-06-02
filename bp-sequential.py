import random
import time

import numpy as np

print('Sequential backpropagation algorithm')


# Функція активації - сигмоїда
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Похідна функції активації
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, learning_rate, regularization_param):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.weights1 = np.random.randn(self.inputs, self.hidden)
        self.weights2 = np.random.randn(self.hidden, self.outputs)

    def feedforward(self, x):
        self.hidden_sum = np.dot(x, self.weights1)
        self.activated_hidden = sigmoid(self.hidden_sum)
        self.output_sum = np.dot(self.activated_hidden, self.weights2)
        self.activated_output = sigmoid(self.output_sum)
        return self.activated_output

    def backward(self, x, y, output):
        self.error = y - output
        self.delta_output = self.error * sigmoid_derivative(output)
        self.error_hidden = self.delta_output.dot(self.weights2.T)
        self.delta_hidden = self.error_hidden * sigmoid_derivative(self.activated_hidden)

        self.weights2 += self.activated_hidden.T.dot(self.delta_output) * self.learning_rate
        self.weights1 += x.T.dot(self.delta_hidden) * self.learning_rate

        # Приміняємо L2-регуляризацію
        self.weights2 -= self.regularization_param * self.weights2
        self.weights1 -= self.regularization_param * self.weights1

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            output = self.feedforward(x)
            self.backward(x, y, output)

    def predict(self, x):
        return self.feedforward(x)


# Тестові дані
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 1000)
y = np.array([[0], [1], [1], [0]] * 1000)
print('Input data shapes:', x.shape, y.shape)

# Визначення параметрів нейромережі
input_neurons = x.shape[1]
output_neurons = y.shape[1]
hidden_neurons = 5

epochs = 20000
learning_rate = 0.001
regularization_param = 0.0001

timer = time.time()

# Ініціалізація та тренування нейромережі
nn = NeuralNetwork(input_neurons, hidden_neurons, output_neurons, learning_rate, regularization_param)
nn.train(x, y, epochs)

print('execution time', int((time.time() - timer) * 1000))

print('weights1', nn.weights1, 'weights2', nn.weights2)

# Тестування навченої нейромережі
print("Predictions:")
print(np.around(nn.predict(x[:4]), 4))
