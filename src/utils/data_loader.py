import gzip
import struct
import urllib.request
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

URLS = {
    "mnist": {
        "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    },
    "fashion_mnist": {
        "train_images": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
        "train_labels": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
        "test_images":  "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
        "test_labels":  "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
    },
}


def get_cache_dir(name):
    path = Path.home() / ".cache" / "da6401" / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def download(url, dest):
    if dest.exists():
        return
    print("Downloading", dest.name)
    urllib.request.urlretrieve(url, dest)

def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)


def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_dataset(name, val_split=0.1):

    name = name.lower().replace("-", "_")
    if name not in URLS:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    urls = URLS[name]
    cache = get_cache_dir(name)

    paths = {}

    for key in urls:
        file_path = cache / Path(urls[key]).name
        download(urls[key], file_path)
        paths[key] = file_path

    X_train_full = read_images(paths["train_images"]).astype(np.float32) / 255.0
    y_train_full = read_labels(paths["train_labels"]).astype(np.int64)

    X_test = read_images(paths["test_images"]).astype(np.float32) / 255.0
    y_test = read_labels(paths["test_labels"]).astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_split,
        random_state=42,
        stratify=y_train_full,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def one_hot(y, num_classes=10):
    result = np.zeros((len(y), num_classes), dtype=np.float32)
    result[np.arange(len(y)), y] = 1
    return result


def get_batches(X, y, batch_size, shuffle=True):

    n = len(X)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, n, batch_size):
        batch_idx = indices[i:i + batch_size]
        yield X[batch_idx], y[batch_idx]