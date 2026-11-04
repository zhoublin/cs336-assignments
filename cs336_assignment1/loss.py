from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def cross_entropy(inputs, targets):
    # 直接计算logsoftmax在分子e^x很小时存在下溢可能 (log 0)
    normalized_inputs = inputs - inputs.max(dim=-1).values.unsqueeze(dim=-1)
    return -(normalized_inputs[torch.arange(normalized_inputs.shape[0]), targets] - \
            torch.log(torch.exp(normalized_inputs).sum(dim=-1))).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, 
        params,
        lr = 1e-3,
        weight_decay = 0,
        betas = (0.9, 0.999),
        eps = 1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            weight_decay = group["weight_decay"]
            (beta1, beta2) = group["betas"]
            epsilon = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data # Get the gradient of loss with respect to p.

                lr_t = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                p.data -= lr * weight_decay * p.data + lr_t * m / (v ** 0.5 + epsilon)

                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v

        return loss

def lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return (
            min_learning_rate + \
            0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * \
            (max_learning_rate - min_learning_rate)
        )
    else:
        return min_learning_rate

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
):
    gradient = torch.tensor(0, dtype=torch.float32)
    for param in parameters:
        if param.grad is None:
            continue
        gradient += torch.pow(param.grad.data, 2.).sum()

    gradient = gradient.pow(0.5)

    if gradient < max_l2_norm:
        return parameters
    else:
        _lambda = max_l2_norm / (gradient + 1e-6)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.data *= _lambda
        return parameters