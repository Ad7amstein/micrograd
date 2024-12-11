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
            _ = Value(data="5")
            _ = Value(data=[3])
            _ = Value(data=[3, 2])
            _ = Value(data={4.2})

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

    def test_mul(self):
        """To be filled"""
        val1 = Value(data=3.2)
        val2 = Value(data=5.9)
        val3 = val1 * val2
        self.assertIsInstance(val3, Value)
        self.assertAlmostEqual(val3.data, 18.88)

        val4 = val1 * 5
        self.assertIsInstance(val4, Value)
        self.assertAlmostEqual(val4.data, 16)
        val5 = val2 * 2.6
        self.assertIsInstance(val5, Value)
        self.assertAlmostEqual(val5.data, 15.34)

        val1 *= 7
        self.assertIsInstance(val1, Value)
        self.assertAlmostEqual(val1.data, 22.4)
        val6 = 1 * val2
        self.assertIsInstance(val6, Value)
        self.assertAlmostEqual(val6.data, 5.9)
        val7 = val2 * 0
        self.assertIsInstance(val7, Value)
        self.assertAlmostEqual(val7.data, 0)
        val8 = -1 * val1
        self.assertIsInstance(val8, Value)
        self.assertAlmostEqual(val8.data, -22.4)

    def test_neg(self):
        """To be filled"""
        val = Value(data=10)
        self.assertEqual(val.data, 10)
        self.assertIsInstance(-val, Value)
        val2 = -val
        self.assertIsNot(val, val2)
        self.assertEqual(val2.data, -10)

    def test_sub(self):
        """To be filled"""
        val1 = Value(data=3.2)
        val2 = Value(data=5.9)
        val3 = val1 - val2
        self.assertIsInstance(val3, Value)
        self.assertAlmostEqual(val3.data, -2.7)
        val3 = val2 - val1
        self.assertIsInstance(val3, Value)
        self.assertAlmostEqual(val3.data, 2.7)

        val4 = val1 - 5
        self.assertIsInstance(val4, Value)
        self.assertAlmostEqual(val4.data, -1.8)
        val5 = val2 - 2.6
        self.assertIsInstance(val5, Value)
        self.assertAlmostEqual(val5.data, 3.3)

        val1 -= 7
        self.assertIsInstance(val1, Value)
        self.assertAlmostEqual(val1.data, -3.8)
        val6 = 0 - val2
        self.assertIsInstance(val6, Value)
        self.assertAlmostEqual(val6.data, -5.9)
        val7 = val2 - 0
        self.assertIsInstance(val7, Value)
        self.assertAlmostEqual(val7.data, 5.9)
        val8 = -1 - val1
        self.assertIsInstance(val8, Value)
        self.assertAlmostEqual(val8.data, 2.8)

    def test_pow(self):
        """To be filled"""
        val1 = Value(data=2)
        val2 = Value(data=3.2)

        self.assertIsInstance(val1**3, Value)
        val3 = val1**3
        self.assertEqual(val3.data, 8)
        self.assertIsInstance(val1**-3, Value)
        val3 = val1**-3
        self.assertEqual(val3.data, 0.125)

        self.assertIsInstance(val1**val2, Value)
        val3 = val1**val2
        self.assertAlmostEqual(val3.data, 9.1895868)
        self.assertIsInstance(val2**val1, Value)
        val3 = val2**val1
        self.assertAlmostEqual(val3.data, 10.24)

        val1 **= 1
        self.assertIsInstance(val1, Value)
        self.assertAlmostEqual(val1.data, 2)
        val2 **= 0
        self.assertIsInstance(val2, Value)
        self.assertAlmostEqual(val2.data, 1)

        val3 = 3**val1
        self.assertIsInstance(val3, Value)
        self.assertAlmostEqual(val3.data, 9)
        val3 = 3**-val1
        self.assertIsInstance(val3, Value)
        self.assertAlmostEqual(val3.data, 0.11111111)
        val3 = -(val1**-1)
        self.assertIsInstance(val3, Value)
        self.assertEqual(val3.data, -0.5)

    def test_div(self):
        """To be filled"""
        val1 = Value(data=7)
        val2 = Value(data=3.2)

        self.assertIsInstance(val1 / 3, Value)
        val3 = val1 / 3
        self.assertAlmostEqual(val3.data, 2.33333333)
        self.assertIsInstance(val1 / -3, Value)
        val3 = val1 / -3
        self.assertAlmostEqual(val3.data, -2.33333333)
        self.assertIsInstance(-val1 / 3, Value)
        val3 = -val1 / 3
        self.assertAlmostEqual(val3.data, -2.33333333)

        self.assertIsInstance(val1 / val2, Value)
        val3 = val1 / val2
        self.assertAlmostEqual(val3.data, 2.1875)
        self.assertIsInstance(val2 / val1, Value)
        val3 = val2 / val1
        self.assertAlmostEqual(val3.data, 0.4571428571)

        val1 /= 1
        self.assertIsInstance(val1, Value)
        self.assertAlmostEqual(val1.data, 7)
        with self.assertRaises(ZeroDivisionError):
            _ = val1 / 0
            _ = 1 / Value(0)
            val2 /= 0

        val3 = 3 / val1
        self.assertIsInstance(val3, Value)
        self.assertAlmostEqual(val3.data, 0.4285714)
        val3 = 0 / val1
        self.assertIsInstance(val3, Value)
        self.assertAlmostEqual(val3.data, 0)
        val3 = 3 / -val1
        self.assertIsInstance(val3, Value)
        self.assertAlmostEqual(val3.data, -0.4285714)
        val3 = -val1 / -1
        self.assertIsInstance(val3, Value)
        self.assertEqual(val3.data, 7)

    def test_children(self):
        """To be filled"""
        val1 = Value(5)
        self.assertIsInstance(val1.prev, set)
        self.assertEqual(val1.prev, set())

        val2 = Value(2)
        val3 = val1 + val2
        self.assertEqual(val3.prev, set((val1, val2)))
        val3 = val1 * val2
        self.assertEqual(val3.prev, set((val1, val2)))
        val3 = val1**val2
        self.assertEqual(val3.prev, set((val1,)))
        val3 = val2**val1
        self.assertEqual(val3.prev, set((val2,)))

    def test_op(self):
        """To be filled"""
        val1 = Value(4)
        self.assertIsInstance(val1.op, str)
        self.assertEqual(val1.op, "")

        val2 = Value(3)
        val3 = val1 + val2
        self.assertEqual(val3.op, "+")
        val3 = val1 - val2
        self.assertEqual(val3.op, "+")
        val3 = val1 * val2
        self.assertEqual(val3.op, "*")
        val3 = val1 / val2
        self.assertEqual(val3.op, "*")
        val3 = val1**val2
        self.assertEqual(val3.op, f"**{val2}")

    def test_backward(self):
        """To be filled"""
        val1 = Value(2)
        self.assertEqual(val1.grad, 0)
        val2 = Value(3)
        self.assertEqual(val2.grad, 0)
        val3 = val1 + val2
        self.assertEqual(val3.grad, 0)
        val3.backward()
        self.assertEqual(val1.grad, 1)
        self.assertEqual(val2.grad, 1)
        self.assertEqual(val3.grad, 1)

        val1 = Value(2)
        val2 = Value(3)
        val3 = val1 * val2
        val3.backward()
        self.assertEqual(val1.grad, 3)
        self.assertEqual(val2.grad, 2)
        self.assertEqual(val3.grad, 1)

        val1 = Value(2)
        val3 = -val1
        val3.backward()
        self.assertEqual(val1.grad, -1)
        self.assertEqual(val3.grad, 1)

        val1 = Value(2)
        val2 = Value(3)
        val3 = val1 - val2
        val3.backward()
        self.assertEqual(val1.grad, 1)
        self.assertEqual(val2.grad, -1)
        self.assertEqual(val3.grad, 1)

        val1 = Value(2)
        val2 = Value(3)
        val3 = val1**val2
        val3.backward()
        self.assertAlmostEqual(val1.grad.data, 3 * (2**2))
        self.assertEqual(val3.grad, 1)

        val1 = Value(2)
        val2 = Value(3)
        val3 = val1 / val2
        val3.backward()
        self.assertAlmostEqual(val1.grad, 1 / 3)
        self.assertEqual(val2.grad.data, -2 * 3**-2)
        self.assertEqual(val3.grad, 1)

        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10)
        e = a * b
        d = e + c
        f = Value(-2.0)
        l = d * f
        l.backward()
        self.assertAlmostEqual(l.grad, 1)
        self.assertAlmostEqual(f.grad, 4)
        self.assertAlmostEqual(d.grad, -2)
        self.assertAlmostEqual(e.grad, -2)
        self.assertAlmostEqual(c.grad, -2)
        self.assertAlmostEqual(a.grad, 6)
        self.assertAlmostEqual(b.grad, -4)
