from my_ml_lib.nn.modules.base import Module
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize weights and biases
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))
        # Store them as parameters for autograd
        self._parameters = {"W": self.W, "b": self.b}

    def call(self, X):
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad_output):
        self.grad_W = self.input.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.W.T
        return grad_input

    def parameters(self):
        return [self.W, self.b]

    def zero_grad(self):
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def __repr__(self):
        return f"Linear({self.W.shape[0]} -> {self.W.shape[1]})"


   
