import numpy as np
from .activations import softmax

def cross_entropy_loss(y_pred_logits, y_true_onehot):
    probs = softmax(y_pred_logits)
    probs = np.clip(probs, 1e-15, 1.0)
    loss = -np.mean(np.sum(y_true_onehot * np.log(probs), axis=1))
    return loss

def cross_entropy_grad(y_pred_logits, y_true_onehot):
    probs = softmax(y_pred_logits)
    batch_size = y_true_onehot.shape[0]
    return (probs - y_true_onehot) / batch_size

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_pred, y_true):
    batch_size = y_true.shape[0]
    output_size = y_true.shape[1]
    return (2.0 / (batch_size * output_size)) * (y_pred - y_true)