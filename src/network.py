import numpy as np

from layer import HiddenLayer, InputLayer, OutputLayer


class Network:

    def __init__(self, layers):
        self.lr = 0.2
        self.momentum = 0.9

        self.input_layer = InputLayer(layers[0])

        self.layers = [self.input_layer]
        prev = self.input_layer
        for x in layers[1:-1]:
            hidden_layer = HiddenLayer(x, prev)
            self.layers.append(hidden_layer)
            prev = hidden_layer
        
        self.output_layer = OutputLayer(layers[-1:], prev)
        self.layers.append(self.output_layer)

    def feedforward(self, x):
        pass

    def backprop(self, result, expected_neurons):
        pass