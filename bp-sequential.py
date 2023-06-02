import numpy as np


# Функція активації - сигмоїда
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Похідна функції активації
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.weights1 = np.random.rand(self.inputs, self.hidden)
        self.weights2 = np.random.rand(self.hidden, self.outputs)

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

        self.weights2 += self.activated_hidden.T.dot(self.delta_output)
        self.weights1 += x.T.dot(self.delta_hidden)

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            output = self.feedforward(x)
            self.backward(x, y, output)

    def predict(self, x):
        return self.feedforward(x)


# Тестові дані
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Визначення параметрів нейромережі
input_neurons = x.shape[1]
output_neurons = y.shape[1]
hidden_neurons = 3

epochs = 10000

# Ініціалізація та тренування нейромережі
nn = NeuralNetwork(input_neurons, hidden_neurons, output_neurons)
nn.train(x, y, epochs)

# Тестування навченої нейромережі
print("Predictions:")
print(nn.predict(x))
