from .base import Module

class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

    def forward(self, x):
        out = x
        for m in self.modules:
            out = m(out)
        return out

    def parameters(self):
        params = []
        for m in self.modules:
            params.extend(m.parameters())
        return params
