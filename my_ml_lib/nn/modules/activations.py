import numpy as np
from .base import Module
 


class Sigmoid(Module):
    def call(self, x):
        # σ(x) = 1 / (1 + exp(-x))
        return (x * (-1)).exp()
 

class ReLU(Module):
    def call(self, X):
        return [[xi.relu() for xi in x] for x in X]

