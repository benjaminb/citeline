from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(ABC, nn.Module):
    registry = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Adapter.registry[cls.__name__] = cls

class BaselineMLPEmbeddingMapper(Adapter):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024):
        super().__init__()
        self.description = f"""2-layer MLP, compressing {input_dim} -> {hidden_dim}, with no skip connection. 
        Using swish activation and minimal weight initializations to prevent model from perturbing vectors too much"""

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.final.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        y = F.silu(self.fc1(x)) # Swish activation
        y = F.silu(self.final(y))
        y = F.normalize(y, p=2.0, dim=1)
        return y

class ResidualEmbeddingMapper(Adapter):
    def __init__(self, input_dim=1024, output_dim=1024, hidden_dim=256):  # Reduced from 512
        super().__init__()
        self.description = f"""2-layer MLP, compressing {input_dim} -> {hidden_dim}, with skip connection to output layer. 
        Using swish activation and minimal weight initializations to prevent model from perturbing vectors too much"""

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim, output_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.final.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        y = F.silu(self.fc1(x))
        y = F.silu(self.final(y))
        y = x + self.residual_scale * y  # skip connection

        y = F.normalize(y, p=2.0, dim=1)
        return y
