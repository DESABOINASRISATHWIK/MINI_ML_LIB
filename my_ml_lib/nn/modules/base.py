import numpy as np

class Module:
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def parameters(self):
        """
        Recursively get all parameters (Value objects) in the module.
        """
        params = []
        for name, param in self._parameters.items():
            params.append(param)
        for name, module in self._modules.items():
            params += module.parameters()
        return params

    def zero_grad(self):
        """
        Reset all gradients to zero.
        """
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError("Forward (call) not implemented")

    def save_state_dict(self, path):
        """
        Save model parameters to a .npz file.
        """
        np.savez(path, **{k: v.data for k, v in self._parameters.items()})

    def load_state_dict(self, path):
        """
        Load model parameters from a .npz file.
        """
        data = np.load(path)
        for k, v in self._parameters.items():
            v.data = data[k]
