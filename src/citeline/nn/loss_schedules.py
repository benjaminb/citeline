from abc import abstractmethod
import torch


class LossSchedule:
    registry = {}

    def __init__(self, total_steps: int):
        self.step = 0
        self.total_steps = total_steps

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        LossSchedule.registry[cls.__name__] = cls

    @abstractmethod
    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    """Returns a tuple of tensors for (positive weights, negative weights)"""

class LinearBasicTriplet(LossSchedule):
    def __init__(self, total_steps: int):
        super().__init__(total_steps=total_steps)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Works with: 1 positive, 1 negative per anchor"""
        pos_weight = torch.tensor(max(0.0, 1 - self.step / self.total_steps))
        neg_weight = torch.tensor(min(1.0, self.step / self.total_steps))
        self.step += 1
        return pos_weight, neg_weight
