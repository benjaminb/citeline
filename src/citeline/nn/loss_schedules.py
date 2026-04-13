from abc import abstractmethod
import torch

"""
Callable: (step: int) -> Tensor[num_pos, dim], Tensor[num_neg, dim]
- step: current training step
"""


def linear_basic_triplet(step: int, total_steps: int):
    """Linearly decays positive weight from 1 to 0, linearly increases negative
    weight from 0 to 1.
    """
    pos_weight = torch.max(torch.tensor(0.0), torch.tensor(1 - step / total_steps)).to("mps")
    neg_weight = torch.min(torch.tensor(1.0), torch.tensor(step / total_steps)).to("mps")
    return pos_weight, neg_weight


class LossSchedule:
    registry = {}

    def __init__(self, step: int, total_steps: int):
        self.step = step
        self.total_steps = total_steps

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        LossSchedule.registry[cls.__name__] = cls

    @abstractmethod
    def __call__(
        self, positives: torch.Tensor["np", "dim"], negatives: torch.Tensor["nn", "dim"]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
