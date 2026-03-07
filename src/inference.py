import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--dataset", default=None,choices=["mnist", "fashion_mnist"])
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--confusion", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config_path) as f:
        config = json.load(f)

    dataset = args.dataset if args.dataset else config.get("dataset", "mnist")
    _, _, (X_test, y_test) = load_dataset(dataset, val_split=args.val_split)

    print("Dataset:", dataset)
    print("Test samples:", len(X_test))

    model = NeuralNetwork(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        output_size=config["output_size"],
        activation=config["activation"],
        weight_init=config["weight_init"],
        loss=config.get("loss", "cross_entropy"),
    )

    model.load(args.model_path)
    print("Model loaded from:", args.model_path)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\nTest Metrics")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))

    if args.confusion:
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix")
        print(cm)


if __name__ == "__main__":
    main()