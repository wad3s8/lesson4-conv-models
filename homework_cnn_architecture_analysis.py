# homework_cnn_architecture_analysis.py

import torch
import time
from torch import nn
from convolutional_basics.datasets import get_cifar_loaders
from utils.training_utils import train_model
from utils.visualization_utils import plot_training_history
from utils.comparison_utils import compare_models, count_parameters
import os
import matplotlib.pyplot as plt


# === МОДЕЛИ ДЛЯ РАЗМЕРОВ ЯДЕР ===
class ConvNetKernel(nn.Module):
    def __init__(self, kernel_size=3, use_1x1=False):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        if use_1x1:
            layers.append(nn.Conv2d(3, 32, 1))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(32, 64, kernel_size, padding=padding))
        else:
            layers.append(nn.Conv2d(3, 64, kernel_size, padding=padding))
        layers += [nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# === МОДЕЛИ ДЛЯ РАЗНОЙ ГЛУБИНЫ ===
def build_deep_cnn(depth=2, residual=False):
    layers = []
    in_channels = 3
    for i in range(depth):
        out_channels = 32 if i < depth - 1 else 64
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if i % 2 == 1:
            layers.append(nn.MaxPool2d(2))
        in_channels = out_channels
    layers.append(nn.AdaptiveAvgPool2d((2, 2)))
    model = nn.Sequential(*layers)
    return nn.Sequential(
        model,
        nn.Flatten(),
        nn.Linear(64 * 2 * 2, 10)
    )


class SimpleResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1)
        )
        self.pool = nn.MaxPool2d(2)
        self.downsample = nn.Conv2d(3, 32, 1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Linear(64 * 2 * 2, 10)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.block1(x)
        x += residual
        x = self.pool(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# === ОСНОВНОЙ ЭКСПЕРИМЕНТ ===
def run_kernel_experiments(device):
    print("\n--- Kernel Size Analysis ---")
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    configs = [(3, False), (5, False), (7, False), (3, True)]
    histories = []
    for ks, use_1x1 in configs:
        model = ConvNetKernel(kernel_size=ks, use_1x1=use_1x1).to(device)
        name = f"{ks}x{ks}" if not use_1x1 else "1x1+3x3"
        print(f"Training kernel: {name}")
        hist = train_model(model, train_loader, test_loader, epochs=10, device=device)
        plot_training_history(hist, save_path=f"results/architecture_analysis/kernel_{name}.png")
        histories.append((name, hist))
    compare_models(histories, save_path="results/architecture_analysis/kernels_comparison.png")


def run_depth_experiments(device):
    print("\n--- CNN Depth Analysis ---")
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    depths = [2, 4, 6]
    histories = []
    for d in depths:
        model = build_deep_cnn(depth=d).to(device)
        print(f"Training CNN with depth={d}")
        hist = train_model(model, train_loader, test_loader, epochs=10, device=device)
        plot_training_history(hist, save_path=f"results/architecture_analysis/depth_{d}.png")
        histories.append((f"depth={d}", hist))

    # Residual модель
    res_model = SimpleResCNN().to(device)
    print("Training Residual CNN")
    res_hist = train_model(res_model, train_loader, test_loader, epochs=10, device=device)
    plot_training_history(res_hist, save_path="results/architecture_analysis/residual_cnn.png")
    histories.append(("Residual", res_hist))

    compare_models(histories, save_path="results/architecture_analysis/depth_comparison.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/architecture_analysis", exist_ok=True)
    run_kernel_experiments(device)
    run_depth_experiments(device)


if __name__ == "__main__":
    main()
