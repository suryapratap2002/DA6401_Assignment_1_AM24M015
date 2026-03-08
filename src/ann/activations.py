import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

ACTIVATIONS = {
    'sigmoid': sigmoid,
    'tanh':    tanh,
    'relu':    relu,
}
ACTIVATION_DERIVATIVES = {
    'sigmoid': sigmoid_derivative,
    'tanh':    tanh_derivative,
    'relu':    relu_derivative,
}
