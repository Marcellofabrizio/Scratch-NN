import numpy as np
from abc import ABCMeta, abstractmethod


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


class InputLayer():

    def __init__(self, num_neurons):
        self.size = num_neurons
        self.z = np.zeros(num_neurons)

    def synapse(self):
        self.next_layer.synapse()

    def calculate_error(self):
        pass

    def update_weights(self, learning_rate, momentum):
        pass


class HiddenLayer():

    def __init__(self, num_neurons, prev_layer):
        print("Creating hidden layer with num_neurons: %d" % num_neurons)
        self.size = num_neurons
        self.prev_layer = prev_layer
        # define a próxima camada para a anterior, no caso a atual.
        self.prev_layer.next_layer = self
        self.z = np.zeros(num_neurons)
        self.w = np.random.randn(num_neurons, prev_layer.size)

    def synapse(self):
        dot_prod = np.dot(self.w, self.prev_layer.z)
        self.z = sigmoid(dot_prod)
        self.next_layer.synapse()

    def calculate_error(self):
        next_layer_error = self.next_layer.error
        error_factor = np.dot(np.transpose(self.next_layer.w), next_layer_error)
        self.error = self.z*(1-self.z)*error_factor

        # chama chamada para correção do erro da camada anterior
        self.prev_layer.calculate_error()

    def update_weights(self, learning_rate, momentum):
        # valor de momento multiplicado com os pesos para
        # como valor para encontrar novo minimo global
        self.w = self.w * momentum
        tmp = np.tile(self.prev_layer.z, (self.error.size, 1))
        tmp = np.dot(np.diag(self.error), tmp)*learning_rate
        self.w = np.add(tmp, self.w)
        self.prev_layer.update_weights(learning_rate, momentum)


class OutputLayer():

    def __init__(self, num_neurons, prev_layer):
        self.size = num_neurons
        self.prev_layer = prev_layer
        # define a próxima camada para a anterior, no caso a atual.
        self.prev_layer.next_layer = self
        self.z = np.zeros(num_neurons)
        self.w = np.random.randn(num_neurons, prev_layer.size)

    def synapse(self):
        dot_prod = np.dot(self.w, self.prev_layer.z)
        self.z = sigmoid(dot_prod)

    def calculate_error(self, expected, learning_rate, momentum):
        self.error = self.z*(1-self.z)*(expected-self.z)
        self.prev_layer.calculate_error()
        self.update_weights(learning_rate, momentum)

    def update_weights(self, learning_rate, momentum):
        self.w = self.w * momentum
        tmp = np.tile(self.prev_layer.z, (self.error.size, 1))
        tmp = np.dot(np.diag(self.error), tmp)*learning_rate
        self.w = np.add(tmp, self.w)
        self.prev_layer.update_weights(learning_rate, momentum)
