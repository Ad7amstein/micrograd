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

    def __repr__(self):
        """To be filled"""
        return f"Value(data={self.data})"

    def __add__(self, other):
        """To be filled"""
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data=self.data + other.data)
        return out

    def __radd__(self, other):
        """To be filled"""
        return self + other
