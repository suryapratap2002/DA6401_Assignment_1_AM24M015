import json
import numpy as np

from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss


class NeuralNetwork:

    def __init__(self, input_size, hidden_sizes, output_size,
                 activation="relu", weight_init="xavier", loss="cross_entropy"):

        self._loss_fn, self._loss_grad = get_loss(loss)

        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = []

        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else "output"
            self.layers.append(
                NeuralLayer(sizes[i], sizes[i+1], activation=act, weight_init=weight_init)
            )

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_pred, y_true, weight_decay=0.0):
        grad = self._loss_grad(y_pred, y_true)

        for layer in reversed(self.layers):
            grad = layer.backward(grad, weight_decay)

    def compute_loss(self, y_pred, y_true, weight_decay=0.0):
        loss = self._loss_fn(y_pred, y_true)

        if weight_decay:
            for layer in self.layers:
                loss += 0.5 * weight_decay * np.sum(layer.W ** 2)

        return float(loss)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def save(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"W{i}"] = layer.W
            params[f"b{i}"] = layer.b
        np.save(path, params)

    def load(self, path):
        params = np.load(path, allow_pickle=True).item()
        for i, layer in enumerate(self.layers):
            layer.W = params[f"W{i}"]
            layer.b = params[f"b{i}"]

    def save_config(self, path):
        config = {"num_layers": len(self.layers)}
        with open(path, "w") as f:
            json.dump(config, f)

    @classmethod
    def from_config(cls, path):
        with open(path) as f:
            cfg = json.load(f)
        return cfg