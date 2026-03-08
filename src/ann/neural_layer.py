import numpy as np
from .activations import ACTIVATIONS, ACTIVATION_DERIVATIVES

class Layer:
    def __init__(self, input_size, output_size, activation='relu', weight_init='xavier'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        if weight_init == 'xavier':
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == 'zeros':
            self.W = np.zeros((input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.input = None
        self.Z = None
        self.A = None

    def forward(self, X):
        self.input = X
        self.Z = X @ self.W + self.b

        if self.activation_name in ACTIVATIONS:
            self.A = ACTIVATIONS[self.activation_name](self.Z)
        else:
            self.A = self.Z  # linear
        return self.A

    def backward(self, dA, weight_decay=0.0):
        batch_size = self.input.shape[0]

        if self.activation_name in ACTIVATION_DERIVATIVES:
            dZ = dA * ACTIVATION_DERIVATIVES[self.activation_name](self.Z)
        else:
            dZ = dA
        self.grad_W = self.input.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = dZ @ self.W.T
        return dA_prev
