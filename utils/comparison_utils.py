# utils/comparison_utils.py

import matplotlib.pyplot as plt


def count_parameters(model):
    """Подсчет числа обучаемых параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models(histories, save_path=None):
    """
    histories: список кортежей (имя_модели, история)
    Каждый history должен быть dict с ключами: train_accs, test_accs, train_losses, test_losses
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories:
        ax1.plot(hist['test_accs'], label=name)
        ax2.plot(hist['test_losses'], label=name)

    ax1.set_title("Test Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Test Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[Saved] Comparison plot: {save_path}")
    else:
        plt.show()
