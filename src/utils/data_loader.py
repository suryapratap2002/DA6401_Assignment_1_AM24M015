import numpy as np

def load_data(dataset='mnist'):
    try:
        from tensorflow.keras.datasets import mnist, fashion_mnist
        if dataset == 'mnist':
            (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
        else:
            (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    except Exception:
        import urllib.request, os
        urls = {
            'mnist': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
            'fashion_mnist': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/fashion-mnist.npz'
        }
        cache_path = f'{dataset}.npz'
        if not os.path.exists(cache_path):
            print(f"Downloading {dataset}...")
            urllib.request.urlretrieve(urls[dataset], cache_path)
        data = np.load(cache_path)
        X_train_full, y_train_full = data['x_train'], data['y_train']
        X_test, y_test = data['x_test'], data['y_test']

    X_train_full = X_train_full.reshape(-1, 784).astype(np.float32) / 255.0
    X_test       = X_test.reshape(-1, 784).astype(np.float32) / 255.0

    val_size         = int(0.1 * len(X_train_full))
    X_val,   y_val   = X_train_full[:val_size], y_train_full[:val_size]
    X_train, y_train = X_train_full[val_size:],  y_train_full[val_size:]

    def to_onehot(y):
        oh = np.zeros((len(y), 10), dtype=np.float32)
        oh[np.arange(len(y)), y] = 1.0
        return oh

    return (X_train, y_train, to_onehot(y_train),
            X_val,   y_val,   to_onehot(y_val),
            X_test,  y_test,  to_onehot(y_test))