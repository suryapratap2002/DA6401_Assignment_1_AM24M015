import numpy as np
from .neural_layer import Layer
from .activations import softmax
from .objective_functions import (
    cross_entropy_loss, cross_entropy_grad,
    mse_loss, mse_grad
)

class NeuralNetwork:
    def __init__(self, input_size_or_config=None, hidden_sizes=None, output_size=None,
                 activation='relu', weight_init='xavier', loss='cross_entropy',
                 layer_sizes=None, optimizer=None, lr=None, weight_decay=0.0):

        if isinstance(input_size_or_config, dict):
            cfg = input_size_or_config
        elif hasattr(input_size_or_config, '__dict__'):
            cfg = vars(input_size_or_config) 
        else:
            cfg = None

        if cfg is not None:
            input_size  = cfg.get('input_size', 784)
            output_size = cfg.get('output_size', 10)
            activation  = cfg.get('activation', 'relu')
            weight_init = cfg.get('weight_init', 'xavier')
            loss        = cfg.get('loss', 'cross_entropy')

            hs = cfg.get('hidden_size', 128)
            nl = cfg.get('num_layers', 3)
            if isinstance(hs, list):
                hidden_sizes = hs
            else:
                hidden_sizes = [hs] * nl


        elif layer_sizes is not None:
            input_size   = layer_sizes[0]
            hidden_sizes = layer_sizes[1:-1]
            output_size  = layer_sizes[-1]

        else:
            input_size = input_size_or_config

        self.loss_name = loss
        self.layers = []

        prev_size = input_size
        for h_size in hidden_sizes:
            self.layers.append(
                Layer(prev_size, h_size, activation=activation, weight_init=weight_init)
            )
            prev_size = h_size
        self.layers.append(
            Layer(prev_size, output_size, activation='linear', weight_init=weight_init)
        )

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        probs = softmax(out)
        return probs, out  

    def predict(self, X):

        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_classes(self, X):
        return self.predict(X)

    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs

    def compute_loss(self, logits, y_onehot):
        if self.loss_name == 'cross_entropy':
            return cross_entropy_loss(logits, y_onehot)
        else:
            return mse_loss(logits, y_onehot)

    def backward(self, logits, y_onehot, weight_decay=0.0):
        if self.loss_name == 'cross_entropy':
            dA = cross_entropy_grad(logits, y_onehot)
        else:
            dA = mse_grad(logits, y_onehot)

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

        if weight_decay > 0.0:
            for layer in self.layers:
                layer.grad_W += weight_decay * layer.W

    def get_weights(self):
        return [(l.W.copy(), l.b.copy()) for l in self.layers]

    def set_weights(self, weights):
        if isinstance(weights, dict):
            weights = [weights[i] for i in sorted(weights.keys())]
        for layer, (W, b) in zip(self.layers, weights):
            layer.W = W.copy()
            layer.b = b.copy()

    def save(self, path):
        weights = {i: (l.W.copy(), l.b.copy()) for i, l in enumerate(self.layers)}
        np.save(path, weights)
        print(f"Model saved to {path}")

    def load(self, path):
        weights = np.load(path, allow_pickle=True).item()
        self.set_weights([weights[i] for i in sorted(weights.keys())])
        print(f"Model loaded from {path}")