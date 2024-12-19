"""Test nn module"""

import unittest
from micrograd.engine import Value
from micrograd.nn import Module, Neuron
from pycodestyle import Checker


class TestNNModule(unittest.TestCase):
    """To be added"""

    def test_doc(self):
        """To be added"""
        self.assertIsNotNone(Module.__doc__)
        self.assertIsNotNone(Module.zero_grad.__doc__)
        self.assertIsNotNone(Module.parameters.__doc__)
        self.assertIsNotNone(Neuron.__doc__)
        self.assertIsNotNone(Neuron.__init__.__doc__)
        self.assertIsNotNone(Neuron.__call__.__doc__)
        self.assertIsNotNone(Neuron.parameters.__doc__)
        self.assertIsNotNone(Neuron.__repr__.__doc__)

    def test_pycodestyle(self):
        """Test the pycodestyle."""
        file = "micrograd/nn.py"
        checker = Checker(file)
        file_errors = checker.check_all()
        self.assertEqual(file_errors, 0)

    def test_module(self):
        """To be added"""
        module = Module()
        self.assertIsInstance(module.parameters(), list)
        self.assertListEqual(module.parameters(), [])

    def test_neuron(self):
        """To be added"""
        node = Neuron(nin=3)
        self.assertIsInstance(node.w, list)
        self.assertIsInstance(node.b, Value)
        self.assertIsInstance(node.act, str)

        self.assertEqual(len(node.w), 3)
        self.assertEqual(node.b.data, 0)
        self.assertIsInstance(node.w[0], Value)
        self.assertIsInstance(node.w[1], Value)
        self.assertIsInstance(node.w[2], Value)
        self.assertEqual(node.act, 'Linear')

        with self.assertRaises(TypeError):
            _ = Neuron(nin=1, activation=3)

        with self.assertRaises(ValueError):
            _ = Neuron(nin=1, activation='adham')

        x = [1, 2, 3]
        val = node(x)
        self.assertIsInstance(val, Value)

        node_params = node.parameters()
        self.assertIsInstance(node_params, list)
        self.assertEqual(len(node_params), 4)
        self.assertIsInstance(node_params[0], Value)
        self.assertIsInstance(node_params[1], Value)
        self.assertIsInstance(node_params[2], Value)
        self.assertIsInstance(node_params[3], Value)

        node_str_repr = repr(node)
        self.assertEqual(node_str_repr, "Neuron(nin=3, activation='Linear')")

        node_eval = eval(node_str_repr)
        self.assertIsInstance(node_eval, Neuron)
        self.assertEqual(len(node_eval.w), 3)
        self.assertEqual(node_eval.b.data, 0)

        node.w[0].grad = 5
        node.w[1].grad = 5
        node.w[2].grad = 5
        node.b.grad = 5
        node.zero_grad()
        node_params_zeros = node.parameters()
        self.assertEqual(node_params_zeros[0].grad, 0)
        self.assertEqual(node_params_zeros[1].grad, 0)
        self.assertEqual(node_params_zeros[2].grad, 0)
        self.assertEqual(node_params_zeros[3].grad, 0)
