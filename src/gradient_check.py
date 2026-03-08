import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ann.neural_network import NeuralNetwork

def compute_loss(model, X, y_oh):
    _, logits = model.forward(X) 
    return model.compute_loss(logits, y_oh)

def numerical_gradient(model, X, y_oh, layer_idx, param='W', eps=1e-5):
    layer = model.layers[layer_idx]
    param_matrix = layer.W if param == 'W' else layer.b
    grad = np.zeros_like(param_matrix)

    it = np.nditer(param_matrix, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original = param_matrix[idx]
        param_matrix[idx] = original + eps
        loss_plus = compute_loss(model, X, y_oh)
        param_matrix[idx] = original - eps
        loss_minus = compute_loss(model, X, y_oh)
        param_matrix[idx] = original  # restore
        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return grad


def relative_error(analytical, numerical):
    numerator   = np.abs(analytical - numerical)
    denominator = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-15)
    return np.max(numerator / denominator)


def max_absolute_error(analytical, numerical):
    return np.max(np.abs(analytical - numerical))

def run_gradient_check(loss='cross_entropy', activation='relu',num_layers=2, hidden_size=4,batch_size=8, input_size=16, output_size=4,seed=42, tolerance=1e-7):

    np.random.seed(seed)
    print(f"\n{'='*60}")
    print(f"  Gradient Check")
    print(f"  loss={loss}  activation={activation} "f"layers={num_layers}  hidden={hidden_size}")
    print(f"  batch={batch_size}  input={input_size}  output={output_size}")
    print(f"  tolerance={tolerance}")
    print(f"{'='*60}")

    X    = np.random.randn(batch_size, input_size) * 0.5
    y_int = np.random.randint(0, output_size, batch_size)
    y_oh  = np.zeros((batch_size, output_size))
    y_oh[np.arange(batch_size), y_int] = 1.0

    model = NeuralNetwork(
        input_size,
        hidden_sizes=[hidden_size] * num_layers,
        output_size=output_size,
        activation=activation,
        weight_init='xavier',
        loss=loss,
    )

    _, logits = model.forward(X)
    model.backward(logits, y_oh, weight_decay=0.0)
    all_passed = True

    for i, layer in enumerate(model.layers):
        for param in ['W', 'b']:
            analytical = layer.grad_W if param == 'W' else layer.grad_b
            numerical  = numerical_gradient(model, X, y_oh, i, param)
            abs_err = max_absolute_error(analytical, numerical)
            rel_err = relative_error(analytical, numerical)
            passed  = abs_err < tolerance
            status = "PASSED" if passed else "FAILED"
            print(f"  Layer {i} grad_{param}  |  "
                  f"abs_err={abs_err:.2e}  rel_err={rel_err:.2e}  "
                  f"[{status}]")

            if not passed:
                all_passed = False
                print(f"    Analytical:\n{analytical}")
                print(f"    Numerical:\n{numerical}")

    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*60}")
    return all_passed

if __name__ == '__main__':
    results = []
    for activation in ['sigmoid', 'tanh', 'relu']:
        for loss in ['cross_entropy', 'mean_squared_error']:
            passed = run_gradient_check(
                loss=loss,
                activation=activation,
                num_layers=2,
                hidden_size=4,
                batch_size=8,
                input_size=16,
                output_size=4,
            )
            results.append((activation, loss, passed))

    passed = run_gradient_check(
        loss='cross_entropy',
        activation='tanh',
        num_layers=4,
        hidden_size=8,
        batch_size=4,
        input_size=10,
        output_size=3,
    )
    results.append(('tanh_deep', 'cross_entropy', passed))

    passed = run_gradient_check(
        loss='cross_entropy',
        activation='relu',
        num_layers=2,
        hidden_size=4,
        batch_size=1,
        input_size=8,
        output_size=3,
    )
    results.append(('relu_bs1', 'cross_entropy', passed))
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for act, loss, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {act:20s}  {loss:20s}  {status}")
        if not passed:
            all_ok = False

    print(f"\n  {'ALL CHECKS PASSED - gradients are correct!' if all_ok else 'SOME CHECKS FAILED - review backward() implementation'}")
    print(f"{'='*60}\n")