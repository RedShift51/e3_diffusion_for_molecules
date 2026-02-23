"""
LoRA (Low-Rank Adaptation) wrapper for linear layers in EGNN.
Adds trainable low-rank matrices to nn.Linear: output += (alpha/rank) * (B @ A) @ x.
"""
from typing import Optional
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Wraps nn.Linear with LoRA: out = linear(x) + (alpha/rank) * (lora_B @ lora_A) @ x."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0,
                 original: Optional[nn.Linear] = None, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features, bias=bias) if original is None else original
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: float = 1.0) -> "LoRALinear":
        out, in_f = linear.weight.shape
        bias = linear.bias is not None
        lora = cls(in_f, out, rank, alpha, original=linear, bias=bias)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        # (B @ A) @ x^T -> (batch, out_features)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        scale = self.alpha / self.rank
        return out + scale * delta


def inject_lora(module: nn.Module, rank: int, alpha: Optional[float] = None) -> int:
    """
    Recursively replace nn.Linear with LoRALinear inside module (e.g. EGNN).
    If alpha is None, use alpha = rank (common default).
    Returns number of layers replaced.
    """
    if alpha is None:
        alpha = float(rank)
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora = LoRALinear.from_linear(child, rank=rank, alpha=alpha)
            setattr(module, name, lora)
            count += 1
        else:
            count += inject_lora(child, rank, alpha)
    return count


def freeze_base_lora(module: nn.Module) -> None:
    """Set requires_grad=False for all parameters except LoRA (lora_A, lora_B)."""
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            if hasattr(child, 'linear') and child.linear is not None:
                for p in child.linear.parameters():
                    p.requires_grad = False
            # lora_A, lora_B stay requires_grad=True (default)
        else:
            freeze_base_lora(child)
