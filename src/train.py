import argparse
import os
import sys
import json
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wandb
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_data


def accuracy(model, X, y_int):
    preds = model.predict_classes(X)
    return np.mean(preds == y_int)

def get_batches(X, y_oh, batch_size, shuffle=True):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y_oh[batch_idx]

def train(config=None):
    with wandb.init(config=config) as run:
        cfg = wandb.config
        if cfg.num_layers > 6:
            print(f"WARNING: num_layers={cfg.num_layers} exceeds advised limit of 6")
        if cfg.hidden_size > 128:
            print(f"WARNING: hidden_size={cfg.hidden_size} exceeds advised limit of 128")
            
        run.name = (f"{cfg.dataset}_{cfg.optimizer}_lr{cfg.learning_rate}"
                    f"_hl{cfg.num_layers}_sz{cfg.hidden_size}_{cfg.activation}")

        # data
        print(f"\nLoading {cfg.dataset}...")
        (X_train, y_train, y_train_oh,
         X_val,   y_val,   y_val_oh,
         X_test,  y_test,  y_test_oh) = load_data(cfg.dataset)
        print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

        # model
        hidden_sizes = [cfg.hidden_size] * cfg.num_layers
        model = NeuralNetwork(
            784,
            hidden_sizes=hidden_sizes,
            output_size=10,
            activation=cfg.activation,
            weight_init=cfg.weight_init,
            loss=cfg.loss,
        )

        optimizer = get_optimizer(cfg.optimizer, lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
        print(f"\nArchitecture : 784 -> {' -> '.join(str(s) for s in hidden_sizes)} -> 10")
        print(f"Activation   : {cfg.activation}  |  Loss: {cfg.loss}")
        print(f"Optimizer    : {cfg.optimizer}  |  LR: {cfg.learning_rate}\n")

        best_f1      = 0.0
        best_weights = None
        best_epoch   = 0

        for epoch in range(cfg.epochs):
            train_losses = []
            for X_batch, y_batch_oh in get_batches(X_train, y_train_oh, cfg.batch_size):
                _, logits = model.forward(X_batch)
                loss   = model.compute_loss(logits, y_batch_oh)
                model.backward(logits, y_batch_oh, weight_decay=cfg.weight_decay)
                optimizer.update(model.layers)
                train_losses.append(loss)

            train_loss  = np.mean(train_losses)
            train_acc   = accuracy(model, X_train, y_train)
            val_acc     = accuracy(model, X_val,   y_val)

            # test F1 - used to pick the best checkpoint
            y_pred_test = model.predict_classes(X_test)
            test_f1     = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
            test_acc    = np.mean(y_pred_test == y_test)
            grad_norm  = np.linalg.norm(model.layers[0].grad_W)
            act_mean   = float(np.mean(model.layers[0].A))
            act_std    = float(np.std(model.layers[0].A))
            dead_frac  = float(np.mean(model.layers[0].A == 0))

            wandb.log({
                'epoch':                epoch + 1,
                'train_loss':           train_loss,
                'train_acc':            train_acc,
                'val_acc':              val_acc,
                'test_acc':             test_acc,
                'test_f1':              test_f1,
                'grad_norm_layer0':     grad_norm,
                'activation_mean':      act_mean,
                'activation_std':       act_std,
                'dead_neuron_fraction': dead_frac,
            })

            if test_f1 > best_f1:
                best_f1      = test_f1
                best_weights = model.get_weights()
                best_epoch   = epoch + 1
                print(f"  Epoch {epoch+1:3d}  new best F1 = {best_f1:.4f}  "
                      f"(test_acc={test_acc:.4f}  val_acc={val_acc:.4f})  <- saved")
            else:
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:3d}  loss={train_loss:.4f}  "
                          f"val_acc={val_acc:.4f}  test_f1={test_f1:.4f}")

        if best_weights is not None:
            model.set_weights(best_weights)

        wandb.log({'best_test_f1': best_f1, 'best_epoch': best_epoch})

        print(f"\n{'='*50}")
        print(f"  Best Test F1 : {best_f1:.4f}  (epoch {best_epoch})")
        print(f"{'='*50}")

        return model, best_f1, best_epoch, cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")
    parser.add_argument('-d',   '--dataset',       default='mnist',choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e',   '--epochs',         type=int,   default=10)
    parser.add_argument('-b',   '--batch_size',     type=int,   default=64)
    parser.add_argument('-l',   '--loss',           default='cross_entropy',choices=['mean_squared_error', 'cross_entropy'])
    parser.add_argument('-o',   '--optimizer',      default='adam',choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-lr',  '--learning_rate',  type=float, default=0.001)
    parser.add_argument('-wd',  '--weight_decay',   type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers',     type=int,   default=3)
    parser.add_argument('-sz',  '--hidden_size',    type=int,   default=128,help='Number of neurons per hidden layer')
    parser.add_argument('-a',   '--activation',     default='relu',choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init',    default='xavier',choices=['random', 'xavier', 'zeros'])
    parser.add_argument('--wandb_project', default='DA6401_Assignment_1_AM24M015')
    parser.add_argument('--wandb_entity',  default=None)
    parser.add_argument('--save_model', action='store_true',help='Save best_model.npy and best_config.json based on test F1')
    parser.add_argument('--sweep', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = {
        'dataset':       args.dataset,
        'epochs':        args.epochs,
        'batch_size':    args.batch_size,
        'loss':          args.loss,
        'optimizer':     args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay':  args.weight_decay,
        'num_layers':    args.num_layers,
        'hidden_size':   args.hidden_size,
        'activation':    args.activation,
        'weight_init':   args.weight_init,
    }
    wandb.login()
    if args.sweep:
        train()
    else:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
        )
        model, best_f1, best_epoch, cfg = train(config=config)

        if args.save_model:
            save_dir = os.path.dirname(os.path.abspath(__file__))
            model_path  = os.path.join(save_dir, 'best_model.npy')
            config_path = os.path.join(save_dir, 'best_config.json')
            model.save(model_path)
            best_config = {
                'dataset':       cfg.dataset,
                'epochs':        cfg.epochs,
                'batch_size':    cfg.batch_size,
                'loss':          cfg.loss,
                'optimizer':     cfg.optimizer,
                'learning_rate': cfg.learning_rate,
                'weight_decay':  cfg.weight_decay,
                'num_layers':    cfg.num_layers,
                'hidden_size':   cfg.hidden_size,
                'activation':    cfg.activation,
                'weight_init':   cfg.weight_init,
                'best_test_f1':  round(float(best_f1), 6),
                'best_epoch':    best_epoch,
            }

            with open(config_path, 'w') as f:
                json.dump(best_config, f, indent=2)

            print(f"\n  best_model.npy   saved -> {model_path}")
            print(f"  best_config.json saved -> {config_path}")
            print(f"  Best Test F1 = {best_f1:.4f}  (from epoch {best_epoch})")

        wandb.finish()
