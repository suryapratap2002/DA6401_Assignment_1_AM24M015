import numpy as np

def cross_entropy_loss(y_pred, y_true):
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


def cross_entropy_grad(y_pred, y_true):
    return y_pred - y_true


def mse_loss(y_pred, y_true):
    return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))


def mse_grad(y_pred, y_true):
    dL_dA = y_pred - y_true

    batch = y_pred.shape[0]
    dZ = np.zeros_like(y_pred)

    for i in range(batch):
        s = y_pred[i].reshape(-1, 1)
        J = np.diagflat(s) - s @ s.T
        dZ[i] = J @ dL_dA[i]

    return dZ


_LOSSES = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_grad),
    "mean_squared_error": (mse_loss, mse_grad),
}


def get_loss(name):
    name = name.lower()
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Supported: {list(_LOSSES.keys())}")
    return _LOSSES[name]