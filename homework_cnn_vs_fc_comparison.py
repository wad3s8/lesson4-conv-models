# homework_cnn_vs_fc_comparison.py

import time
import torch
from models.cnn_models import SimpleCNN, CNNWithResidual, CIFARCNN
from models.fc_models import FullyConnectedNet
from utils.training_utils import train_model
from utils.visualization_utils import plot_training_history
from utils.comparison_utils import count_parameters, compare_models
from convolutional_basics.datasets import get_mnist_loaders, get_cifar_loaders


def run_mnist_experiments(device):
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    print("\n--- MNIST Experiments ---")

    fc_net = FullyConnectedNet(input_size=28*28, num_classes=10).to(device)
    print(f"FC Network parameters: {count_parameters(fc_net)}")
    print("Training FC Network...")
    fc_history = train_model(fc_net, train_loader, test_loader, epochs=10, device=device)
    plot_training_history(fc_history, save_path="results/mnist_comparison/fc_training.png")

    simple_cnn = SimpleCNN(input_channels=1, num_classes=10).to(device)
    print(f"Simple CNN parameters: {count_parameters(simple_cnn)}")
    print("Training Simple CNN...")
    simple_history = train_model(simple_cnn, train_loader, test_loader, epochs=10, device=device)
    plot_training_history(simple_history, save_path="results/mnist_comparison/simplecnn_training.png")

    residual_cnn = CNNWithResidual(input_channels=1, num_classes=10).to(device)
    print(f"Residual CNN parameters: {count_parameters(residual_cnn)}")
    print("Training Residual CNN...")
    residual_history = train_model(residual_cnn, train_loader, test_loader, epochs=10, device=device)
    plot_training_history(residual_history, save_path="results/mnist_comparison/rescnn_training.png")

    compare_models([
        ("FC", fc_history),
        ("SimpleCNN", simple_history),
        ("ResCNN", residual_history)
    ], save_path="results/mnist_comparison/accuracy_vs_loss.png")


def run_cifar_experiments(device):
    train_loader, test_loader = get_cifar_loaders(batch_size=64)

    print("\n--- CIFAR-10 Experiments ---")

    cifar_cnn = CIFARCNN(num_classes=10).to(device)
    print(f"CIFAR CNN parameters: {count_parameters(cifar_cnn)}")
    print("Training CIFAR CNN...")
    cifar_history = train_model(cifar_cnn, train_loader, test_loader, epochs=10, device=device)
    plot_training_history(cifar_history, save_path="results/cifar_comparison/simplecnn_training.png")

    residual_cnn = CNNWithResidual(input_channels=3, num_classes=10).to(device)
    print(f"Residual CNN parameters: {count_parameters(residual_cnn)}")
    print("Training Residual CNN...")
    residual_history = train_model(residual_cnn, train_loader, test_loader, epochs=10, device=device)
    plot_training_history(residual_history, save_path="results/cifar_comparison/rescnn_training.png")

    compare_models([
        ("SimpleCNN", cifar_history),
        ("ResCNN", residual_history)
    ], save_path="results/cifar_comparison/accuracy_vs_loss.png")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    run_mnist_experiments(device)
    run_cifar_experiments(device)


if __name__ == '__main__':
    main()
