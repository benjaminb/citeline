"""
QueryMapper architectures.

Two variants:
  - "mlp"      : Simple feedforward MLP with GELU activations and dropout.
  - "residual" : MLP where layers of matching width use additive residual connections
                 (like a lightweight ResNet). Good when hidden_dims are all the same size.

Only query vectors pass through the model; positive/negative chunk vectors stay fixed.
Output is L2-normalized so cosine similarity == dot product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from citeline.nn.contrastive.config import ModelConfig


class MLP(nn.Module):
    """Plain MLP: input_dim → h1 → h2 → ... → input_dim, L2-normalized output."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        dims = [input_dim] + hidden_dims + [input_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no activation/dropout on final layer
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ResidualBlock(nn.Module):
    """Two-layer block with residual connection. Requires in_dim == out_dim."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class ResidualMLP(nn.Module):
    """
    MLP with residual connections.

    Architecture:
      - Linear projection: input_dim → hidden_dims[0]
      - N ResidualBlocks at hidden_dims[0] width (all hidden dims must be equal)
      - Linear projection: hidden_dims[-1] → input_dim
      - L2-normalized output
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        if len(set(hidden_dims)) != 1:
            raise ValueError(
                f"ResidualMLP requires all hidden_dims to be equal, got {hidden_dims}. "
                "Use arch='mlp' for variable-width layers."
            )
        super().__init__()
        hidden_dim = hidden_dims[0]
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in hidden_dims])
        self.proj_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.proj_in(x))
        for block in self.blocks:
            h = block(h)
        return F.normalize(self.proj_out(h), dim=-1)


def build_model(input_dim: int, cfg: ModelConfig) -> nn.Module:
    """Factory: instantiate the right architecture from ModelConfig."""
    if cfg.arch == "mlp":
        return MLP(input_dim, cfg.hidden_dims, cfg.dropout)
    elif cfg.arch == "residual":
        return ResidualMLP(input_dim, cfg.hidden_dims, cfg.dropout)
    else:
        raise ValueError(f"Unknown arch: {cfg.arch!r}. Choose 'mlp' or 'residual'.")
