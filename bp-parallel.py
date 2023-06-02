import os
from mpi4py import MPI
import numpy as np

os.environ['OMP_NUM_THREADS'] = '4'  # Set the number of threads to 4


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

    def feedforward(self, X):
        self.hidden_sum = np.dot(X, self.weights1)
        self.activated_hidden = sigmoid(self.hidden_sum)
        self.output_sum = np.dot(self.activated_hidden, self.weights2)
        self.activated_output = sigmoid(self.output_sum)
        return self.activated_output

    def backward(self, X, y, output):
        self.error = y - output
        self.delta_output = self.error * sigmoid_derivative(output)
        self.error_hidden = self.delta_output.dot(self.weights2.T)
        self.delta_hidden = self.error_hidden * sigmoid_derivative(self.activated_hidden)

        self.weights2 += self.activated_hidden.T.dot(self.delta_output)
        self.weights1 += X.T.dot(self.delta_hidden)

    def train(self, X, y, epochs, comm):
        for epoch in range(epochs):
            output = self.feedforward(X)

            local_error = np.mean((output - y) ** 2)
            global_error = comm.allreduce(local_error, op=MPI.SUM)
            if global_error < 1e-5:
                break

            self.backward(X, y, output)

    def predict(self, X):
        return self.feedforward(X)


# Тестові дані
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Визначення параметрів нейромережі
input_neurons = x.shape[1]
output_neurons = y.shape[1]
hidden_neurons = 3
epochs = 10000

# Ініціалізація MPI комунікатора
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('rank:', rank, 'size:', size)

# Розподіл даних між процесами
local_x = np.array_split(x, size)[rank]
local_y = np.array_split(y, size)[rank]

# Ініціалізація та тренування нейромережі на кожному процесі
nn = NeuralNetwork(input_neurons, hidden_neurons, output_neurons)
nn.train(local_x, local_y, epochs, comm)

# Збір результатів навчання на кореневому процесі
trained_weights1 = comm.gather(nn.weights1, root=0)
trained_weights2 = comm.gather(nn.weights2, root=0)

# Виведення результатів на кореневому процесі
if rank == 0:
    final_weights1 = np.vstack(trained_weights1)
    final_weights2 = np.vstack(trained_weights2)

    nn.weights1 = final_weights1
    nn.weights2 = final_weights2

    # Тестування навченої нейромережі
    print("Predictions:")
    print(nn.predict(x))
