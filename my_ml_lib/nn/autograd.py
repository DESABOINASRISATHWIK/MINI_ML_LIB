import numpy as np

class Value:
    """
    A scalar or tensor value that supports automatic differentiation.
    """

    def __init__(self, data, _children=(), op="", label=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self.prev = set(_children)
        self.op = op
        self.label = label

    # ----- Representation -----
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self.op})"

    # ----- Core Ops -----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # handles other + self
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        out = Value(-self.data, (self,), "neg")

        def _backward():
            self.grad -= out.grad
        out._backward = _backward
        return out

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

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __pow__(self, power):
        assert np.isscalar(power)
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad
        out._backward = _backward
        return out

    # ----- Matrix Multiplication -----
    def matmul(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data.dot(other.data), (self, other), "matmul")

        def _backward():
            self.grad += out.grad.dot(other.data.T)
            other.grad += self.data.T.dot(out.grad)
        out._backward = _backward
        return out

    # ----- Activations -----
    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data + 1e-8), (self,), "log")

        def _backward():
            self.grad += (1 / (self.data + 1e-8)) * out.grad
        out._backward = _backward
        return out

    # ----- Reductions -----
    def sum(self, axis=None, keepdims=False):
        out = Value(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            self.grad += out.grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Value(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")

        def _backward():
            n = self.data.size if axis is None else self.data.shape[axis]
            self.grad += (out.grad * np.ones_like(self.data)) / n
        out._backward = _backward
        return out

    # ----- Backprop -----
    def backward(self):
        """
        Backpropagate gradients through the graph from this node (root).
        """
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()


# ----- Graph Traversal for Visualization -----
def get_all_nodes_and_edges(root):
    """
    Traverse backward from a root Value node (usually the loss)
    to collect all nodes and edges in the computation graph.
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for parent in getattr(v, "prev", []):
                edges.add((parent, v))
                build(parent)

    build(root)
    return nodes, edges



