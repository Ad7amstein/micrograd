"""To be filled"""


class Value:
    """To be filled"""

    def __init__(self, data):
        """To be filled"""
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None

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

    def __add__(self, other):
        """To be filled"""
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data=self.data + other.data)
        return out

    def __mul__(self, other):
        """To be filled"""
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data=self.data * other.data)
        return out

    def __pow__(self, other):
        """To be filled"""
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data=self.data**other.data)
        return out

    def __rpow__(self, other):
        """To be filled"""
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data=other.data**self.data)
        return out

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
