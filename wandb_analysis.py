import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools
import wandb
from sklearn.metrics import confusion_matrix
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_dataset, one_hot, get_batches


def build_model(hidden, activation="relu", weight_init="xavier", loss="cross_entropy"):
    return NeuralNetwork(784, hidden, 10,activation=activation,weight_init=weight_init,loss=loss)

def quick_train(model, optimizer, Xtr, ytr, Xval, yval_oh, yval,
                epochs=5, batch_size=64, log_grad=False, log_act=False):

    history = []
    for ep in range(epochs):
        for xb, yb in get_batches(Xtr, ytr, batch_size):
            yp = model.forward(xb)
            model.backward(yp, yb)
            optimizer.update(model.layers)

        yp = model.forward(Xval)
        val_loss = model.compute_loss(yp, yval_oh)
        val_acc = (np.argmax(yp, 1) == yval).mean()

        grad_norm = None
        if log_grad:
            grad_norm = np.linalg.norm(model.layers[0].grad_W)

        acts = None
        if log_act:
            xs, _ = next(get_batches(Xtr, ytr, 256, shuffle=False))
            model.forward(xs)
            acts = model.layers[0].A.copy()

        history.append({
            "epoch": ep + 1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "grad": grad_norm,
            "acts": acts
        })

    return history

def plot_optimizer_comparison(run, Xtr, ytr, Xval, yval_oh, yval):

    opts = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for opt_name in opts:
        lr = 0.01 if opt_name in ["sgd", "momentum", "nag"] else 0.001
        model = build_model([128, 128, 128])
        opt = get_optimizer(opt_name, lr=lr)
        hist = quick_train(model, opt, Xtr, ytr, Xval, yval_oh, yval)
        ep = [h["epoch"] for h in hist]
        ax[0].plot(ep, [h["val_loss"] for h in hist], label=opt_name)
        ax[1].plot(ep, [h["val_acc"] for h in hist], label=opt_name)

    ax[0].set_title("Validation Loss")
    ax[1].set_title("Validation Accuracy")

    for a in ax:
        a.set_xlabel("Epoch")
        a.legend()
        a.grid(True)
    plt.tight_layout()
    run.log({"optimizer_comparison": wandb.Image(fig)})
    plt.close()


def log_confusion(run, model, Xtest, ytest):
    y_pred = model.predict(Xtest)
    cm = confusion_matrix(ytest, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i, j in itertools.product(range(10), range(10)):
        ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    run.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", default="da6401-assignment1")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--best_model", default=None)
    parser.add_argument("--best_config", default=None)

    args = parser.parse_args()
    np.random.seed(42)
    (Xtr, ytr), (Xval, yval), (Xtest, ytest) = load_dataset(args.dataset)

    ytr_oh = one_hot(ytr)
    yval_oh = one_hot(yval)
    X_small = Xtr[:10000]
    y_small = ytr_oh[:10000]

    with wandb.init(project=args.wandb_project,
                    entity=args.wandb_entity,
                    name="analysis") as run:

        plot_optimizer_comparison(run, X_small, y_small, Xval, yval_oh, yval)

        if args.best_model and args.best_config:
            model = NeuralNetwork.from_config(args.best_config)
            model.load(args.best_model)
            log_confusion(run, model, Xtest, ytest)

    print("Analysis finished.")

if __name__ == "__main__":
    main()