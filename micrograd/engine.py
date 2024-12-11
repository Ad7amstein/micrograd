"""To be filled"""


class Value:
    """To be filled"""

    def __init__(self, data, _children=(), _op=""):
        """To be filled"""
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.prev = set(_children)
        self.op = _op

    @property
    def data(self):
        """To be filled"""
        return self.__data

    @data.setter
    def data(self, data):
        """To be filled"""
        if not (isinstance(data, int) or isinstance(data, float)):
            raise TypeError("Data must be of type: int or float")
        self.__data = data

    @property
    def prev(self):
        """To be filled"""
        return self.__prev

    @prev.setter
    def prev(self, children):
        """To be filled"""
        self.__prev = children

    @property
    def op(self):
        """To be filled"""
        return self.__op

    @op.setter
    def op(self, _op):
        """To be filled"""
        self.__op = _op

    def __add__(self, other):
        """To be filled"""
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """To be filled"""
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __pow__(self, other):
        """To be filled"""
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data**other.data, (self,), f"**{other}")

        def _backward():
            self.grad = (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """To be filled"""
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
        """To be filled"""
        other = other if isinstance(other, Value) else Value(data=other)
        return other**self

    def __neg__(self):
        """To be filled"""
        return self * -1

    def __sub__(self, other):
        """To be filled"""
        return self + (-other)

    def __radd__(self, other):
        """To be filled"""
        return self + other

    def __rmul__(self, other):
        """To be filled"""
        return self * other

    def __rsub__(self, other):
        """To be filled"""
        return other + (-self)

    def __truediv__(self, other):
        """To be filled"""
        denominator = 1
        if isinstance(other, int) or isinstance(other, float):
            denominator = other
        if isinstance(other, Value):
            denominator = other.data
        if denominator == 0:
            raise ZeroDivisionError("Can't divide by zero.")
        return self * (other**-1)

    def __rtruediv__(self, other):
        """To be filled"""
        denominator = self.data
        if denominator == 0:
            raise ZeroDivisionError("Can't divide by zero.")
        return other * (self**-1)

    def __repr__(self):
        """To be filled"""
        return f"Value(data={self.data})"
