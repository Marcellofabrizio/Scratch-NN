import numpy as np

from layer import HiddenLayer, InputLayer, OutputLayer


class Network:

    def __init__(self, layers, lr, momentum):
        self.lr = lr
        self.momentum = momentum

        self.input_layer = InputLayer(layers[0])

        self.layers = [self.input_layer]
        prev = self.input_layer
        for x in layers[1:-1]:
            hidden_layer = HiddenLayer(x, prev)
            self.layers.append(hidden_layer)
            prev = hidden_layer
        
        self.output_layer = OutputLayer(layers[-1:][0], prev)
        self.layers.append(self.output_layer)

    def feedforward(self, x):
        self.input_layer.z = x
        self.input_layer.synapse()
        return np.argmax(self.output_layer.z)

    def backprop(self, expected_neuron):
        expected_output = np.zeros(self.output_layer.size)
        expected_output[expected_neuron] = 1
        # print("Expected output: ", expected_output)
        self.output_layer.calculate_error(expected_output, self.lr, self.momentum)