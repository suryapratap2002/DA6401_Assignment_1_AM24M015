import numpy as np
from ann.activations import get_activation, softmax


class NeuralLayer:

    def __init__(self, in_features, out_features, activation="relu", weight_init="xavier"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.weight_init = weight_init

        self.W, self.b = self._init_weights()

        if activation == "output":
            self._act_fn = None
            self._act_grad = None
        else:
            self._act_fn, self._act_grad = get_activation(activation)

        self.Z = None
        self.A = None
        self.A_prev = None

        self.grad_W = None
        self.grad_b = None

    def _init_weights(self):
        if self.weight_init == "xavier":
            std = np.sqrt(2 / (self.in_features + self.out_features))
            W = np.random.randn(self.in_features, self.out_features) * std
        
        else:
            W = np.random.randn(self.in_features, self.out_features) * 0.01

        b = np.zeros((1, self.out_features))
        return W, b

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b

        if self.activation_name == "output":
            self.A = softmax(self.Z)
        else:
            self.A = self._act_fn(self.Z)

        return self.A

    def backward(self, dA, weight_decay=0.0):
        batch_size = self.A_prev.shape[0]

        if self.activation_name == "output":
            dZ = dA
        else:
            dZ = dA * self._act_grad(self.Z)

        self.grad_W = (self.A_prev.T @ dZ) / batch_size + weight_decay * self.W
        self.grad_b = np.mean(dZ, axis=0, keepdims=True)

        dA_prev = dZ @ self.W.T
        return dA_prev

    def get_params(self):
        return {"W": self.W, "b": self.b}

    def set_params(self, params):
        self.W = params["W"]
        self.b = params["b"]