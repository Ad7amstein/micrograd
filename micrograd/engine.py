"""
This is the main file for the micrograd engine,
which is a simple autograd engine for educational purposes.
"""

import math


class Value:
    """Value in the computation graph"""

    def __init__(self, data, _children=(), _op=""):
        """initialization

        Args:
            data (int or float): data of the value
            _children (tuple, optional): children nodes. Defaults to ().
            _op (str, optional): operation. Defaults to "".
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.prev = set(_children)
        self.op = _op

    @property
    def data(self):
        """Get the data of the value.

        Returns:
            int or float: data of the value
        """
        return self.__data

    @data.setter
    def data(self, data):
        """Set the data of the value.

        Args:
            data (int or float): data of the value

        Raises:
            TypeError: Data must be of type: int or float
        """
        if not (isinstance(data, int) or isinstance(data, float)):
            raise TypeError("Data must be of type: int or float")
        self.__data = data

    @property
    def prev(self):
        """Get the children nodes.

        Returns:
            set: children nodes
        """
        return self.__prev

    @prev.setter
    def prev(self, children):
        """Set the children nodes.

        Args:
            children (tuple): children nodes
        """
        self.__prev = children

    @property
    def op(self):
        """Get the operation.

        Returns:
            str: operation
        """
        return self.__op

    @op.setter
    def op(self, _op):
        """Set the operation.

        Args:
            _op (str): operation
        """
        self.__op = _op

    def tanh(self):
        """Hyperbolic tangent function.

        Returns:
            Value: result of the tanh operation
        """
        x = self.data
        t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """Rectified linear unit function.

        Returns:
            Value: result of the ReLU operation
        """
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        """Define addition operation.

        Args:
            other (Value or int or float): other value

        Returns:
            Value: result of the addition
        """
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """Define multiplication operation.

        Args:
            other (Value or int or float): other value

        Returns:
            Value: result of the multiplication
        """
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __pow__(self, other):
        """Define power operation.

        Args:
            other (Value or int or float): other value

        Returns:
            Value: result of the power operation
        """
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data**other.data, (self,), f"**{other}")

        def _backward():
            self.grad = (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """Backward pass."""
        topo_list = []
        visited = set()

        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    topo_sort(child)
                topo_list.append(v)

        topo_sort(self)

        self.grad = 1
        for v in reversed(topo_list):
            v._backward()

    def __rpow__(self, other):
        """Define power operation for the other value.

        Args:
            other (Value or int or float): other value

        Returns:
            Value: result of the power operation
        """
        other = other if isinstance(other, Value) else Value(data=other)
        return other**self

    def __neg__(self):
        """Define negation operation.

        Returns:
            Value: negation of the value
        """
        return self * -1

    def __sub__(self, other):
        """Subtract another value from this value.

        Args:
            other (Value or numeric): The value to subtract.

        Returns:
            Value: The result of the subtraction.
        """
        return self + (-other)

    def __radd__(self, other):
        """Add this value to another value (right-hand side).

        Args:
            other (Value or numeric): The value to add.

        Returns:
            Value: The result of the addition.
        """
        return self + other

    def __rmul__(self, other):
        """Multiply this value by another value (right-hand side).

        Args:
            other (Value or numeric): The value to multiply.

        Returns:
            Value: The result of the multiplication.
        """
        return self * other

    def __rsub__(self, other):
        """Subtract this value from another value (right-hand side).

        Args:
            other (Value or numeric): The value to subtract from.

        Returns:
            Value: The result of the subtraction.
        """
        return other + (-self)

    def __truediv__(self, other):
        """Divide this value by another value.

        Args:
            other (Value or numeric): The value to divide by.

        Returns:
            Value: The result of the division.

        Raises:
            ZeroDivisionError: If the other value is zero.
        """
        denominator = 1
        if isinstance(other, int) or isinstance(other, float):
            denominator = other
        if isinstance(other, Value):
            denominator = other.data
        if denominator == 0:
            raise ZeroDivisionError("Can't divide by zero.")
        return self * (other**-1)

    def __rtruediv__(self, other):
        """Divide another value by this value (right-hand side).

        Args:
            other (Value or numeric): The value to be divided.

        Returns:
            Value: The result of the division.

        Raises:
            ZeroDivisionError: If this value is zero.
        """
        denominator = self.data
        if denominator == 0:
            raise ZeroDivisionError("Can't divide by zero.")
        return other * (self**-1)

    def __repr__(self):
        """Representation of the value.

        Returns:
            str: representation of the value
        """
        return f"Value(data={self.data})"
