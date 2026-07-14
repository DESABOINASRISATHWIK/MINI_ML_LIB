class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer for autograd-enabled models.
    Works with Value objects that have .data and .grad attributes.
    """

    def __init__(self, params, lr=0.01):
        """
        Parameters
        ----------
        params : list
            List of model parameters (Value objects).
        lr : float
            Learning rate (default = 0.01).
        """
        self.params = params
        self.lr = lr

    def zero_grad(self):
        """Reset gradients for all parameters to zero."""
        for p in self.params:
            p.grad = 0.0

    def step(self):
        """Perform one optimization step."""
        for p in self.params:
            p.data -= self.lr * p.grad


