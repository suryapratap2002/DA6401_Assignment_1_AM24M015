# DA6401 Assignment 1

## Multi-Layer Perceptron for Image Classification

**GitHub Repository:** -- "https://github.com/suryapratap2002/DA6401_Assignment_1_AM24M015"

**Weights & Biases Report:** 

------------------------------------------------------------------------

# Repository Structure

    da6401_assignment_1/
    │
    ├── README.md
    ├── requirements.txt
    │
    ├── models/                       # Saved models
    │   └── .gitkeep
    │
    └── src/
        ├── train.py                  # Training entry point
        ├── inference.py              # Evaluation / inference
        ├── sweep.py                  # W&B hyperparameter sweep
        ├── wandb_analysis.py         # Generates plots for report
    │
        ├── ann/
        │   ├── __init__.py
        │   ├── activations.py        # sigmoid | tanh | relu | softmax
        │   ├── neural_layer.py       # Dense layer implementation
        │   ├── neural_network.py     # MLP model class
        │   ├── objective_functions.py# loss functions
        │   └── optimizers.py         # SGD | Momentum | NAG | RMSProp | Adam | Nadam
    │
        └── utils/
            ├── __init__.py
            └── data_loader.py        # dataset loading and batching

------------------------------------------------------------------------

# Installation

Install the required Python dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Training

Run training from the **repository root directory**.

### Example (MNIST)

``` bash
python src/train.py -d mnist -e 20 -b 64 -l cross_entropy -o adam -lr 0.001 -wd 0.0005 -nhl 3 -sz 128 128 128 -a relu -w_i xavier
```

### With Weights & Biases Logging

``` bash
python src/train.py -d mnist -e 20 -o adam -lr 0.001 --wandb --wandb_project da6401-assignment1
```

### Fashion-MNIST Example

``` bash
python src/train.py -d fashion_mnist -e 20 -o adam -lr 0.001 -nhl 3 -sz 128 -a relu -w_i xavier
```

------------------------------------------------------------------------

# Command Line Arguments
```bash
  ---------------------------------------------------------------------------------------
  Short Flag      Long Flag         Default         Description      Possible Values
  --------------- ----------------- --------------- ---------------- --------------------
  -d              --dataset         mnist           Dataset used for mnist, fashion_mnist
                                                    training         

  -e              --epochs          10              Number of        integer
                                                    training epochs  

  -b              --batch_size      64              Training batch   integer
                                                    size             

  -l              --loss            cross_entropy   Loss function    cross_entropy,
                                                                     mean_squared_error

  -o              --optimizer       adam            Optimization     sgd, momentum, nag,
                                                    algorithm        rmsprop, adam, nadam

  -lr             --learning_rate   1e-3            Learning rate    float

  -wd             --weight_decay    0.0             L2               float
                                                    regularization   
                                                    coefficient      

  -nhl            --num_layers      3               Number of hidden integer
                                                    layers           

  -sz             --hidden_size     \[128\]         Hidden layer     integer or list
                                                    size(s)          

  -a              --activation      relu            Activation       sigmoid, tanh, relu
                                                    function         

  -w_i            --weight_init     xavier          Weight           random, xavier
                                                    initialization   
  ---------------------------------------------------------------------------------------
```
### Saved Outputs

After training, the following files are saved:

    src/best_model.npy
    src/best_config.json

------------------------------------------------------------------------

# Inference / Evaluation

To evaluate a trained model:

``` bash
python src/inference.py --model_path models/best_model.npy --config_path models/best_config.json --confusion
```

This prints:
``` bash
-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix
```
------------------------------------------------------------------------

# Numerical Gradient Check

To verify the correctness of the backpropagation implementation:

``` bash
python src/gradient_check.py
```

This compares analytical gradients from backpropagation with numerical
gradients using central difference.

Tolerance used: **1e-7**

------------------------------------------------------------------------

# Implementation Notes

## Layer Design

Each neural layer exposes gradients:

    self.grad_W
    self.grad_b

These values are computed during:

    NeuralLayer.backward()

------------------------------------------------------------------------

## Gradient Derivation

### Cross-Entropy + Softmax

    dL/dZ = y_pred − y_true

### MSE + Softmax

The gradient uses the softmax Jacobian:

    J = diag(s) − s·sᵀ

------------------------------------------------------------------------

# Supported Optimizers
``` bash
  Optimizer   Hyperparameters
  ----------- -----------------------------------------------
  SGD         learning_rate
  Momentum    learning_rate, β = 0.9
  NAG         learning_rate, β = 0.9
  RMSProp     learning_rate, β = 0.9, ε = 1e-8
  Adam        learning_rate, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
  Nadam       learning_rate, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
```
------------------------------------------------------------------------

# Datasets
``` bash
  Dataset         Classes                  Image Size
  --------------- ------------------------ ------------
  MNIST           10 digits                28 × 28
  Fashion-MNIST   10 clothing categories   28 × 28
```
