import math

class Value:
    """
    A scalar value that supports automatic differentiation.
    Similar to a simplified PyTorch Tensor with requires_grad=True.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data                # actual numeric value
        self.grad = 0.0                 # accumulated gradient
        self._backward = lambda: None   # function to compute local gradient
        self._prev = set(_children)     # parent Values (inputs)
        self._op = _op                  # operation name (for graph)
        self.label = label              # optional name

    # ---------------- Arithmetic Operations ----------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # handles (number + Value)
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supports int/float powers"
        out = Value(self.data ** power, (self,), f"**{power}")
        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # ---------------- Backpropagation ----------------

    def backward(self):
        # topological sort of the computation graph
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # ---------------- Debug Helpers ----------------
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
