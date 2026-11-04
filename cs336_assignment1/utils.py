import os
import matplotlib.pyplot as plt
from typing import Sequence


def plot_loss(
    losses: Sequence[float],
    save_path: str,
    title: str = "Training Loss",
):
    """
    Plot training loss curve and save it.

    Args:
        losses: Loss values recorded at each training iteration.
        save_path: Path to save the loss curve image.
        title: Plot title.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    steps = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linewidth=1.2, label="Training Loss")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()