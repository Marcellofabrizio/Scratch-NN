import numpy as np


class Network:

    def __init__(self, layers):
        self.lr = 0.5
        self.layers = layers
        self.n_layers = len(layers)
        self.biases = [np.random.randn(1, y) for y in layers[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(layers[:-1], layers[1:])]
        self.errors = []
        self.zs = []

    def feedforward(self, x):
        result = x
        for w, b in zip(self.weights, self.biases):
            dot_prod = np.dot(result, w)+b
            result = sigmoid(dot_prod)
            self.zs.append(result)

        return result

    def backprop(self, result, expected_neurons):
        error = result*(1-result)*(expected_neurons-result)
        self.errors.append(error)
        for w, z in zip(self.weights[1:], self.zs[:1]):
            error_factor = error.dot(np.transpose(w))
            error = z*(1-z)*error_factor
            self.errors.append(error)

        for i in range(len(self.weights)-1,1,-1):
            print(len(self.errors))
            self.weights[i] = self.weights[i-1] + self.lr*self.zs[i-1] + self.errors[i]
            print(self.weights[i].shape)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
