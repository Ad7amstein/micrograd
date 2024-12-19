"""To be filled"""

import random
from micrograd.engine import Value


class Module:
    """To be added"""

    def zero_grad(self):
        """To be added"""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """To be added"""
        return []


class Neuron(Module):
    """To be added"""

    def __init__(self, nin, activation="Linear"):
        """To be added"""
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.act = activation

    @property
    def act(self):
        """To be added"""
        return self.__act

    @act.setter
    def act(self, act):
        """To be added"""
        if not isinstance(act, str):
            raise TypeError("Activation must be of type: str")
        if act not in ["Linear"]:
            raise ValueError(
                "Not supported activation, Current available activations are:\
                \n (Linear,)"
            )
        self.__act = act

    def __call__(self, x):
        """To be added"""
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out

    def parameters(self):
        """To be added"""
        return self.w + [self.b]

    def __repr__(self):
        """To be added"""
        return f"Neuron(nin={len(self.w)}, activation='{self.act}')"
