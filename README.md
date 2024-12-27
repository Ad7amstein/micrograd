# micrograd

A tiny scalar-only autograd engine (with a PyTorch-like interface) that you can use to understand the basics of autograd engines.

This project is essentialy a replication of the PyTorch autograd engine. It is a tiny scalar-only autograd engine that you can use to understand the basics of autograd engines.

The project is inspired by the [micrograd project](https://github.com/karpathy/micrograd/blob/master/demo.ipynb)

## Project structure

```bash

micrograd/
├── micrograd
│   ├── __init__.py
│   ├── nn.py
│   ├── engine.py
test/
├── __init__.py
├── test_nn.py
├── test_engine.py

```

## How to use

```python
from micrograd.nn import MLP

# create a simple MLP
model = MLP(1, [10, 5], act='relu')

# forward pass
x = np.array([0.1, 0.2])
y = model(x)

# backward pass
loss = loss_fn(x, y) # must be defined earlier
loss.backward()

# update weights
for p in model.parameters():
    p.data -= learning_rate * p.grad

```

See the `demo.ipynb` notebook for more examples.

## How to run the tests

```bash
python -m unittest discover tests
```
