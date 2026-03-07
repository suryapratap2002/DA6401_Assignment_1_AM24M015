import argparse
import types
import wandb

from train import train


sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [10, 15, 20]},
        "batch_size": {"values": [32, 64, 128]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4,"max": 1e-1, },
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] },
        "num_layers": {"values": [2, 3, 4, 5,6]},
        "hidden_size": {"values": [32, 64, 128]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "weight_init": {"values": ["random", "xavier"]},
        "weight_decay": {"distribution": "log_uniform_values","min": 1e-5,"max": 1e-2, },
        "loss": {"values": ["cross_entropy", "mean_squared_error"]},
    },
}

def run_sweep():
    with wandb.init():
        cfg = wandb.config
        args = types.SimpleNamespace(
            dataset="mnist",
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            loss=cfg.loss,
            optimizer=cfg.optimizer,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            num_layers=cfg.num_layers,
            hidden_size=[cfg.hidden_size],
            activation=cfg.activation,
            weight_init=cfg.weight_init,
            wandb=True,
            wandb_project=wandb.run.project,
            wandb_entity=wandb.run.entity,
            run_name=wandb.run.name,
            save_dir=None,
            val_split=0.1,
            seed=42,
        )
        train(args)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", default="da6401-assignment1")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--dataset", default="mnist")

    return parser.parse_args()


def main():

    args = parse_args()
    sweep_config["parameters"]["dataset"] = {"value": args.dataset}
    sweep_id = wandb.sweep(
        sweep_config,
        project=args.wandb_project,
        entity=args.wandb_entity,
    )

    print("Sweep ID:", sweep_id)
    wandb.agent(sweep_id, function=run_sweep, count=args.count)


if __name__ == "__main__":
    main()