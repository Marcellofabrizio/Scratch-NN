import numpy as np
from abc import ABCMeta, abstractmethod

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class Layer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def summation_unit(self):
        pass

    @abstractmethod
    def transfer_unit(self):
        pass

class InputLayer(Layer):

    def __init__(self, size):
        self.size = size
        self.z = np.zeros(size)

    def synapse(self):
        self.next_layer.synapse()