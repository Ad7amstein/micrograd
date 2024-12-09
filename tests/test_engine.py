"""To be filled"""

import unittest
from micrograd.engine import Value
from pycodestyle import Checker


class TestEngine(unittest.TestCase):
    """To be filled"""

    def test_doc(self):
        """To be filled"""

    def test_pycodestyle(self):
        """Test the pycodestyle."""
        file = "micrograd/engine.py"
        checker = Checker(file)
        file_errors = checker.check_all()
        self.assertEqual(file_errors, 0)

    def test_init(self):
        """To be filled"""
        val = Value(data=5)
        self.assertIsInstance(val, Value)
        self.assertEqual(val.data, 5)
        self.assertEqual(val.grad, 0)

    def test_data(self):
        """To be filled"""
        with self.assertRaises(TypeError):
            Value(data="5")
            Value(data=[3])
            Value(data=[3, 2])
            Value(data={4.2})

    def test_repr(self):
        """To be filled"""
        val1 = Value(data=2.9)
        str_repr1 = repr(val1)
        self.assertIsInstance(str_repr1, str)
        self.assertEqual(str_repr1, "Value(data=2.9)")

        val2 = eval(str_repr1)
        self.assertIsInstance(val2, Value)
        self.assertEqual(val2.data, 2.9)
        str_repr2 = repr(val2)
        self.assertIsInstance(str_repr2, str)
        self.assertEqual(str_repr2, "Value(data=2.9)")

        self.assertEqual(str_repr1, str_repr2)
        self.assertIsNot(val1, val2)

    def test_add(self):
        """To be filled"""
        val1 = Value(data=3.2)
        val2 = Value(data=5.9)
        val3 = val1 + val2
        self.assertIsInstance(val3, Value)
        self.assertAlmostEqual(val3.data, 9.1)

        val4 = val1 + 5
        self.assertIsInstance(val4, Value)
        self.assertAlmostEqual(val4.data, 8.2)
        val5 = val2 + 2.6
        self.assertIsInstance(val5, Value)
        self.assertAlmostEqual(val5.data, 8.5)

        val1 += 7
        self.assertIsInstance(val1, Value)
        self.assertAlmostEqual(val1.data, 10.2)
        val6 = 1 + val2
        self.assertIsInstance(val6, Value)
        self.assertAlmostEqual(val6.data, 6.9)
