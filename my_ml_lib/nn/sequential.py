from my_ml_lib.nn.modules.base import Module

class Sequential(Module):
    """
    Sequential container to hold and run layers one after another.
    Similar to torch.nn.Sequential.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def call(self, X):
        """
        Forward pass through all layers sequentially.
        """
        for layer in self.layers:
            X = layer(X)
        return X

    def parameters(self):
        """
        Collects parameters from all submodules.
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def zero_grad(self):
        """
        Sets gradients of all parameters to zero.
        """
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()

    def __repr__(self):
        """
        Pretty string representation for printing.
        """
        layer_strs = [repr(layer) for layer in self.layers]
        return f"Sequential(\n  " + "\n  ".join(layer_strs) + "\n)"
