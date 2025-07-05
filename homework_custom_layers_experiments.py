# homework_custom_layers_experiments.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from convolutional_basics.datasets import get_cifar_loaders
from utils.training_utils import train_model
from utils.visualization_utils import plot_training_history
from utils.comparison_utils import compare_models, count_parameters
import os


# === Кастомные слои ===
class CustomConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.conv2d(x, self.weight, self.bias, padding=1)
        out = self.bn(out)
        return F.relu(out + 0.1 * torch.sin(out))  # дополнительная логика


class CustomActivation(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.relu(x))


class CustomPooling(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 2) + 0.1 * F.max_pool2d(x, 2)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.fc2 = nn.Linear(in_channels // 4, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_pool)))).view(b, c, 1, 1)
        return x * weights


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = CustomConvLayer(3, 32, 3)
        self.pool = CustomPooling()
        self.attn = ChannelAttention(32)
        self.layer2 = nn.Conv2d(32, 64, 3, padding=1)
        self.act = CustomActivation()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.attn(x)
        x = self.layer2(x)
        x = self.act(x)
        return self.head(x)


# === Residual блоки ===
class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        bottleneck = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, bottleneck, 1)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.conv3 = nn.Conv2d(bottleneck, in_channels, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return F.relu(out + x)


class WideResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        width = in_channels * 2
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, in_channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)


class ResNetVariant(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.block = block(64)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 10)
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.block(x)
        return self.head(x)


# === Запуск экспериментов ===
def run_custom_layer_experiment(device):
    print("\n--- Custom Layers Experiment ---")
    train_loader, test_loader = get_cifar_loaders()
    model = CustomCNN().to(device)
    hist = train_model(model, train_loader, test_loader, epochs=10, device=device)
    plot_training_history(hist, save_path="results/architecture_analysis/custom_cnn.png")


def run_residual_variants_experiment(device):
    print("\n--- Residual Variants Experiment ---")
    train_loader, test_loader = get_cifar_loaders()
    blocks = [BasicResidualBlock, BottleneckResidualBlock, WideResidualBlock]
    names = ["Basic", "Bottleneck", "Wide"]
    histories = []
    for block, name in zip(blocks, names):
        model = ResNetVariant(block).to(device)
        print(f"Training {name} ResNet variant")
        hist = train_model(model, train_loader, test_loader, epochs=10, device=device)
        plot_training_history(hist, save_path=f"results/architecture_analysis/resblock_{name}.png")
        histories.append((name, hist))
    compare_models(histories, save_path="results/architecture_analysis/resblocks_comparison.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/architecture_analysis", exist_ok=True)
    run_custom_layer_experiment(device)
    run_residual_variants_experiment(device)


if __name__ == "__main__":
    main()