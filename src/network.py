import numpy as np


class Network:

    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        self.biases = [np.random.randn(y,1) for y in layers[1:]]
        self.weights = [np.random.randn(y ,x) 
                for x,y in zip(layers[:-1], layers[1:])]

    def feedforward(self, x):
        result = x
        for w,b in zip(self.weights,self.biases):
            dot_prod = np.dot(w,result)+b
            result = sigmoid(dot_prod)
        return result

    def backprop(self, z):
        print(z)
        for w in reversed(self.weights):
            z = np.dot(z, w)
            print(z)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
