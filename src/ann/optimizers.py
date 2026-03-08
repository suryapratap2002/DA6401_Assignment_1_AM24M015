import numpy as np

class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class Momentum:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.vW, self.vb = {}, {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.vW:
                self.vW[i] = np.zeros_like(layer.W)
                self.vb[i] = np.zeros_like(layer.b)
            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * layer.grad_b
            layer.W -= self.lr * self.vW[i]
            layer.b -= self.lr * self.vb[i]


class NAG:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.vW, self.vb = {}, {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.vW:
                self.vW[i] = np.zeros_like(layer.W)
                self.vb[i] = np.zeros_like(layer.b)
            vW_prev = self.vW[i].copy()
            vb_prev = self.vb[i].copy()
            self.vW[i] = self.beta * self.vW[i] + self.lr * layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + self.lr * layer.grad_b
            layer.W -= (1 + self.beta) * self.vW[i] - self.beta * vW_prev
            layer.b -= (1 + self.beta) * self.vb[i] - self.beta * vb_prev


class RMSprop:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.sW, self.sb = {}, {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.sW:
                self.sW[i] = np.zeros_like(layer.W)
                self.sb[i] = np.zeros_like(layer.b)
            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * layer.grad_W ** 2
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * layer.grad_b ** 2
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.sW[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.sb[i]) + self.eps)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.mW, self.mb, self.vW, self.vb = {}, {}, {}, {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if i not in self.mW:
                self.mW[i] = np.zeros_like(layer.W)
                self.mb[i] = np.zeros_like(layer.b)
                self.vW[i] = np.zeros_like(layer.W)
                self.vb[i] = np.zeros_like(layer.b)
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * layer.grad_W ** 2
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * layer.grad_b ** 2
            mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
            mb_hat = self.mb[i] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.t)
            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


class Nadam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.mW, self.mb, self.vW, self.vb = {}, {}, {}, {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        b1t = self.beta1 ** self.t
        b2t = self.beta2 ** self.t
        for i, layer in enumerate(layers):
            if i not in self.mW:
                self.mW[i] = np.zeros_like(layer.W)
                self.mb[i] = np.zeros_like(layer.b)
                self.vW[i] = np.zeros_like(layer.W)
                self.vb[i] = np.zeros_like(layer.b)
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * layer.grad_W ** 2
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * layer.grad_b ** 2
            vW_hat = self.vW[i] / (1 - b2t)
            vb_hat = self.vb[i] / (1 - b2t)
            mW_nes = (self.beta1 * self.mW[i] / (1 - b1t * self.beta1)
                      + (1 - self.beta1) * layer.grad_W / (1 - b1t))
            mb_nes = (self.beta1 * self.mb[i] / (1 - b1t * self.beta1)
                      + (1 - self.beta1) * layer.grad_b / (1 - b1t))
            layer.W -= self.lr * mW_nes / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_nes / (np.sqrt(vb_hat) + self.eps)


def get_optimizer(name, lr, weight_decay=0.0):
    name = name.lower()
    if name == 'sgd':
        return SGD(lr=lr, weight_decay=weight_decay)
    elif name == 'momentum':
        return Momentum(lr=lr, weight_decay=weight_decay)
    elif name == 'nag':
        return NAG(lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return RMSprop(lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return Adam(lr=lr, weight_decay=weight_decay)
    elif name == 'nadam':
        return Nadam(lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from sgd, momentum, nag, rmsprop, adam, nadam")
