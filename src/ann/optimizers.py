import numpy as np


class SGD:

    def __init__(self, lr=0.01, **kwargs):
        self.lr = lr

    def update(self, layers, weight_decay=0.0):
        for layer in layers:
            if layer.grad_W is None:
                continue

            layer.W -= self.lr * (layer.grad_W + weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b


class Momentum:

    def __init__(self, lr=0.01, beta=0.9, **kwargs):
        self.lr = lr
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def update(self, layers, weight_decay=0.0):
        for idx, layer in enumerate(layers):

            if layer.grad_W is None:
                continue

            if idx not in self.vW:
                self.vW[idx] = np.zeros_like(layer.W)
                self.vb[idx] = np.zeros_like(layer.b)

            self.vW[idx] = self.beta * self.vW[idx] + layer.grad_W
            self.vb[idx] = self.beta * self.vb[idx] + layer.grad_b

            layer.W -= self.lr * (self.vW[idx] + weight_decay * layer.W)
            layer.b -= self.lr * self.vb[idx]


class NAG:

    def __init__(self, lr=0.01, beta=0.9, **kwargs):
        self.lr = lr
        self.beta = beta
        self.vW = {}
        self.vb = {}
        self.prevW = {}
        self.prevb = {}

    def apply_lookahead(self, layers):

        for i, layer in enumerate(layers):

            self.prevW[i] = layer.W.copy()
            self.prevb[i] = layer.b.copy()

            vW = self.vW.get(i, np.zeros_like(layer.W))
            vb = self.vb.get(i, np.zeros_like(layer.b))

            layer.W -= self.beta * vW
            layer.b -= self.beta * vb

    def update(self, layers, weight_decay=0.0):

        for i, layer in enumerate(layers):

            if layer.grad_W is None:
                continue

            if i in self.prevW:
                layer.W = self.prevW[i]
                layer.b = self.prevb[i]

            if i not in self.vW:
                self.vW[i] = np.zeros_like(layer.W)
                self.vb[i] = np.zeros_like(layer.b)

            self.vW[i] = self.beta * self.vW[i] + layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + layer.grad_b

            layer.W -= self.lr * (self.vW[i] + weight_decay * layer.W)
            layer.b -= self.lr * self.vb[i]


class RMSProp:

    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, **kwargs):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sW = {}
        self.sb = {}

    def update(self, layers, weight_decay=0.0):

        for i, layer in enumerate(layers):

            if layer.grad_W is None:
                continue

            if i not in self.sW:
                self.sW[i] = np.zeros_like(layer.W)
                self.sb[i] = np.zeros_like(layer.b)

            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * (layer.grad_W ** 2)
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * (layer.grad_b ** 2)

            layer.W -= self.lr * (
                layer.grad_W / (np.sqrt(self.sW[i]) + self.eps) + weight_decay * layer.W
            )

            layer.b -= self.lr * (
                layer.grad_b / (np.sqrt(self.sb[i]) + self.eps)
            )


class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mW = {}
        self.mb = {}
        self.vW = {}
        self.vb = {}
        self.step = 0

    def update(self, layers, weight_decay=0.0):

        self.step += 1

        for i, layer in enumerate(layers):

            if layer.grad_W is None:
                continue

            if i not in self.mW:
                self.mW[i] = np.zeros_like(layer.W)
                self.mb[i] = np.zeros_like(layer.b)
                self.vW[i] = np.zeros_like(layer.W)
                self.vb[i] = np.zeros_like(layer.b)

            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b

            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (layer.grad_W ** 2)
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (layer.grad_b ** 2)

            mW_hat = self.mW[i] / (1 - self.beta1 ** self.step)
            mb_hat = self.mb[i] / (1 - self.beta1 ** self.step)

            vW_hat = self.vW[i] / (1 - self.beta2 ** self.step)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.step)

            layer.W -= self.lr * (
                mW_hat / (np.sqrt(vW_hat) + self.eps) + weight_decay * layer.W
            )

            layer.b -= self.lr * (
                mb_hat / (np.sqrt(vb_hat) + self.eps)
            )


class Nadam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mW = {}
        self.mb = {}
        self.vW = {}
        self.vb = {}
        self.step = 0

    def update(self, layers, weight_decay=0.0):

        self.step += 1

        for i, layer in enumerate(layers):

            if layer.grad_W is None:
                continue

            if i not in self.mW:
                self.mW[i] = np.zeros_like(layer.W)
                self.mb[i] = np.zeros_like(layer.b)
                self.vW[i] = np.zeros_like(layer.W)
                self.vb[i] = np.zeros_like(layer.b)

            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b

            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (layer.grad_W ** 2)
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (layer.grad_b ** 2)

            vW_hat = self.vW[i] / (1 - self.beta2 ** self.step)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.step)

            mW_n = (
                self.beta1 * self.mW[i] / (1 - self.beta1 ** (self.step + 1))
                + (1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** self.step)
            )

            mb_n = (
                self.beta1 * self.mb[i] / (1 - self.beta1 ** (self.step + 1))
                + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self.step)
            )

            layer.W -= self.lr * (
                mW_n / (np.sqrt(vW_hat) + self.eps) + weight_decay * layer.W
            )

            layer.b -= self.lr * (
                mb_n / (np.sqrt(vb_hat) + self.eps)
            )


def get_optimizer(name, **kwargs):

    name = name.lower()

    optimizers = {
        "sgd": SGD,
        "momentum": Momentum,
        "nag": NAG,
        "rmsprop": RMSProp,
        "adam": Adam,
        "nadam": Nadam,
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer '{name}'")

    return optimizers[name](**kwargs)