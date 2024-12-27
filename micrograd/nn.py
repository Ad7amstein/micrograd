"""Neural network modules"""

import random
from micrograd.engine import Value


class Module:
    """base class for all neural network modules"""

    def zero_grad(self):
        """zero gradients for all parameters"""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """return all parameters

        Returns:
            list: list of parameters
        """
        return []


class Neuron(Module):
    """Represents a single neuron"""

    def __init__(self, nin, activation="Linear"):
        """Initialize neuron

        Args:
            nin (int): number of input neurons
            activation (str, optional): activation function.
                                        Defaults to "Linear".
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.act = activation

    @property
    def act(self):
        """Return the activation function

        Returns:
            str: activation function
        """
        return self.__act

    @act.setter
    def act(self, act):
        """Set the activation function

        Args:
            act (str): activation function

        Raises:
            TypeError: Activation must be of type: str
            ValueError: Not supported activation,
                        Current available activations are: (Linear,)
        """
        if not isinstance(act, str):
            raise TypeError("Activation must be of type: str")
        if act not in ["Linear", "relu", "tanh"]:
            raise ValueError(
                "Not supported activation, Current available activations are:\
                \n (Linear, relu, tanh)"
            )
        self.__act = act

    def __call__(self, x):
        """Forward pass of the neuron

        Args:
            x (list): input values

        Returns:
            Value: output value
        """
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.act == "relu":
            return out.relu()
        elif self.act == "tanh":
            return out.tanh()
        else:
            return out

    def parameters(self):
        """Return all parameters

        Returns:
            list: list of parameters
        """
        return self.w + [self.b]

    def __repr__(self):
        """Return string representation of the neuron

        Returns:
            str: string representation
        """
        return f"Neuron(nin={len(self.w)}, activation='{self.act}')"


class Layer(Module):
    """Represents a layer of neurons"""

    def __init__(self, nin, nout, act="Linear"):
        """Initialize layer

        Args:
            nin (int): number of input neurons
            nout (int): number of output neurons
        """
        self.neurons = [Neuron(nin, act) for _ in range(nout)]

    def __call__(self, x):
        """Forward pass of the layer

        Args:
            x (list): input values

        Returns:
            Value: output value
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """Return all parameters

        Returns:
            list: list of parameters
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """Return string representation of the layer

        Returns:
            str: string representation
        """
        return (
            f"Layer(nin={len(self.neurons[0].parameters())-1},"
            f" nout={len(self.neurons)})"
        )


class MLP(Module):
    """To be filled"""

    def __init__(self, nin: int, nouts: list, act: str = "Linear"):
        """To be filled"""
        sz = [nin] + nouts
        self.layers = [
            Layer(
                nin=sz[i],
                nout=sz[i + 1],
                act=act) for i in range(len(sz) - 2)
        ]
        self.layers.append(Layer(nin=sz[len(sz)-2], nout=sz[len(sz)-1], act='Linear'))
        # sz = [nin] + nouts
        # self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        """To be filled"""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """To be filled"""
        return [p for layer in self.layers for p in layer.parameters()]

    def __str__(self):
        """To be filled"""
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
