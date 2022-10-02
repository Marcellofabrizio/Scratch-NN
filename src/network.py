import numpy as np


class Network:

    def __init__(self, layers):
        self.lr = 0.5
        self.layers = layers
        self.n_layers = len(layers)
        self.biases = [np.random.randn(1,y) for y in layers[1:]]
        self.weights = [np.random.randn(x,y) 
                for x,y in zip(layers[:-1], layers[1:])]
        self.errors = []
        self.zs = []

    def feedforward(self, x):
        result = x
        for w,b in zip(self.weights,self.biases):
            dot_prod = np.dot(result,w)+b
            result = sigmoid(dot_prod)
            self.zs.append(result)

        return result

    def backprop(self, z):
        self.errors.append(z)
        # print(z)
        for w,r in zip(reversed(self.weights[1:]), self.zs):
            z = np.dot(z, np.transpose(w))
            error = r*(1-r)*z
            self.errors.append(error)

        for w,r,e in zip(self.weights, self.zs,reversed(self.errors)):
            w = w + self.lr*r*e
            # print(w.shape)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
