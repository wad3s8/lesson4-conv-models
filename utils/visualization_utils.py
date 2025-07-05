import os
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI (для headless сохранения)
import matplotlib.pyplot as plt


def plot_training_history(history, save_path=None):
    """Визуализирует историю обучения и сохраняет график"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved] Plot to: {os.path.abspath(save_path)}")
    else:
        print("[Warning] No save_path provided and GUI disabled (Agg backend)")
