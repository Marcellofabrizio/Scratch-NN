import numpy as np


class Network:

    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        self.weights = [np.random.randn(y ,x) 
                for x,y in zip(layers[:-1], layers[1:])]

    def feedforward(self, x):
        result = x
        for w in self.weights:
            dot_prod = np.dot(w, result)
            result = sigmoid(result)
        return result

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
