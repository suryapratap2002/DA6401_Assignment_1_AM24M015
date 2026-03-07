import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)  # stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


_ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
}


def get_activation(name):
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Supported: {list(_ACTIVATIONS.keys())}")
    return _ACTIVATIONS[name]