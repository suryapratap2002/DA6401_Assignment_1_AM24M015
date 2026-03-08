"""
inference.py  -  Load saved weights and evaluate on test set.

Usage:
    python inference.py -d mnist -m best_model.npy -c best_config.json
    python inference.py -d mnist --model_path best_model.npy --config best_config.json
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a saved MLP model")
    parser.add_argument('-d', '--dataset', default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    # support both --model_path and -m
    parser.add_argument('-m', '--model_path', '--model', default='best_model.npy',
                        dest='model_path', help='Path to .npy weights file')
    # support both --config and -c
    parser.add_argument('-c', '--config', default='best_config.json',
                        help='Path to best_config.json')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── load config ───────────────────────────────────────────────────────────
    config_path = args.config if os.path.isabs(args.config) else \
        os.path.join(os.path.dirname(__file__), args.config)

    if not os.path.exists(config_path):
        # try relative to cwd
        config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(config_path) as f:
        cfg = json.load(f)

    print("Config loaded:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # ── load data ─────────────────────────────────────────────────────────────
    dataset = cfg.get('dataset', args.dataset)
    print(f"\nLoading dataset: {dataset}")
    (_, _, _,
     _, _, _,
     X_test, y_test, _) = load_data(dataset)

    # ── build model ───────────────────────────────────────────────────────────
    model = NeuralNetwork(cfg)   # passes config dict directly

    # ── load weights ─────────────────────────────────────────────────────────
    model_path = args.model_path if os.path.isabs(args.model_path) else \
        os.path.join(os.path.dirname(__file__), args.model_path)

    if not os.path.exists(model_path):
        model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model.load(model_path)

    # ── predict ───────────────────────────────────────────────────────────────
    y_pred = model.predict_classes(X_test)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n" + "="*45)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}  (macro)")
    print(f"  Recall    : {recall:.4f}  (macro)")
    print(f"  F1-Score  : {f1:.4f}  (macro)")
    print("="*45)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return acc, precision, recall, f1


if __name__ == '__main__':
    main()