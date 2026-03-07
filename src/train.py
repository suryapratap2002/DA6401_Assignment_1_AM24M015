import argparse
import json
import os
import sys
import numpy as np
from sklearn.metrics import f1_score

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_dataset, one_hot, get_batches

def parse_args():
    parser = argparse.ArgumentParser("NumPy MLP Trainer")
    parser.add_argument("--dataset", default="mnist",choices=["mnist", "fashion_mnist"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--loss", default="cross_entropy",choices=["cross_entropy", "mean_squared_error"])
    parser.add_argument("--optimizer", default="adam",choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, nargs="+", default=[128])
    parser.add_argument("--activation", default="relu",choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("--weight_init", default="xavier",choices=["random", "xavier"])
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--grad_check", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="da6401-assignment1")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default=None)

    return parser.parse_args()

def resolve_save_dir(path):
    if path:
        return path
    repo_root = os.path.dirname(SRC_DIR)
    return os.path.join(repo_root, "models")


def build_hidden_sizes(num_layers, hidden_size):
    if len(hidden_size) == 1:
        return hidden_size * num_layers
    if len(hidden_size) == num_layers:
        return hidden_size
    raise ValueError("hidden_size must be either 1 value or num_layers values")

def evaluate(model, X, y_oh, y_int, weight_decay=0.0):
    probs = model.forward(X)
    loss = model.compute_loss(probs, y_oh, weight_decay)
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == y_int))
    f1 = float(f1_score(y_int, preds, average="macro", zero_division=0))

    return loss, acc, f1
def grad_norm(layer):
    if layer.grad_W is None:
        return 0.0
    return float(np.sqrt(np.sum(layer.grad_W ** 2) + np.sum(layer.grad_b ** 2)))

def grad_check(model, X, y, eps=1e-5, tol=1e-7):

    print("\nRunning gradient check...")

    X = X.astype(np.float64)
    y = y.astype(np.float64)

    pred = model.forward(X)
    model.backward(pred, y)
    analytical = [{"W": l.grad_W.copy(), "b": l.grad_b.copy()} for l in model.layers]
    success = True

    for li, layer in enumerate(model.layers):
        for name, param in [("W", layer.W), ("b", layer.b)]:
            num_grad = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"])

            while not it.finished:
                idx = it.multi_index
                orig = param[idx]
                param[idx] = orig + eps
                loss1 = model.compute_loss(model.forward(X), y)
                param[idx] = orig - eps
                loss2 = model.compute_loss(model.forward(X), y)
                param[idx] = orig
                num_grad[idx] = (loss1 - loss2) / (2 * eps)
                it.iternext()

            err = np.max(np.abs(analytical[li][name] - num_grad))
            status = "OK" if err < tol else "FAIL"

            if err >= tol:
                success = False

            print(f"Layer {li} {name}: error={err:.2e} [{status}]")

    print("Gradient check:", "PASSED" if success else "FAILED")
    return success

def train(args):

    np.random.seed(args.seed)
    save_dir = resolve_save_dir(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.npy")
    best_config_path = os.path.join(save_dir, "best_config.json")
    print(f"\nLoading dataset: {args.dataset}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        load_dataset(args.dataset, val_split=args.val_split)
    y_train_oh = one_hot(y_train)
    y_val_oh = one_hot(y_val)
    y_test_oh = one_hot(y_test)

    print("Train:", X_train.shape[0],
          "Val:", X_val.shape[0],
          "Test:", X_test.shape[0])

    hidden_sizes = build_hidden_sizes(args.num_layers, args.hidden_size)

    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=hidden_sizes,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init,
        loss=args.loss
    )

    print("Network:", [784] + hidden_sizes + [10])

    # gradient check mode
    if args.grad_check:
        grad_check(model, X_train[:8], y_train_oh[:8])
        return model, {}

    optimizer = get_optimizer(args.optimizer, lr=args.learning_rate)
    best_val_f1 = -1

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):

        total_loss = 0
        batches = 0

        for Xb, yb in get_batches(X_train, y_train_oh, args.batch_size):

            pred = model.forward(Xb)
            loss = model.compute_loss(pred, yb, args.weight_decay)
            model.backward(pred, yb, weight_decay=args.weight_decay)
            optimizer.update(model.layers, weight_decay=args.weight_decay)
            total_loss += loss
            batches += 1

        train_loss = total_loss / batches
        val_loss, val_acc, val_f1 = evaluate(
            model, X_val, y_val_oh, y_val, args.weight_decay
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save(best_model_path)

            cfg = {
                "hidden_sizes": hidden_sizes,
                "activation": args.activation,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            }
            with open(best_config_path, "w") as f:
                json.dump(cfg, f, indent=2)

    model.load(best_model_path)
    test_loss, test_acc, test_f1 = evaluate(model, X_test, y_test_oh, y_test)
    print("\nTest results")
    print("Accuracy :", test_acc)
    print("F1 score :", test_f1)
    print("\nSaved model:", best_model_path)
    return model, {"accuracy": test_acc, "f1": test_f1}

if __name__ == "__main__":
    args = parse_args()
    train(args)