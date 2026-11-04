import os
import torch
import numpy as np
from typing import BinaryIO, IO

def data_loading(
    dataset: np.typing.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    seq_length = dataset.shape[0]
    
    # 采样
    indices = np.random.randint(0, seq_length - context_length, (batch_size,))
    offsets = np.arange(context_length)
    x_seq = dataset[indices[:, None] + offsets[None, :]]

    indices += 1
    y_seq = dataset[indices[:, None] + offsets[None, :]]

    x_seq = torch.from_numpy(x_seq).to(device)
    y_seq = torch.from_numpy(y_seq).to(device)
    return (x_seq, y_seq)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    info = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(info, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    info = torch.load(src)
    model.load_state_dict(info['model'])
    optimizer.load_state_dict(info['optimizer'])
    return info['iteration']