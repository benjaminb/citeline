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
        """Works with: 1 positive, 1 negative per anchor
        Interpolates between (pos_weight=1.0, neg_weight=0.0) at step 0 and (pos_weight=0.5, neg_weight=0.5) at step total_steps"""
        neg_proportion = 0.5 * min(1.0, self.step / self.total_steps)
        pos_weight = torch.tensor(max(0.0, 1 - neg_proportion))
        neg_weight = torch.tensor(neg_proportion)
        self.step += 1
        return pos_weight, neg_weight

class LinearSlow(LossSchedule):

    def __init__(self, total_steps: int):
        super().__init__(total_steps=total_steps)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Works with: 1 positive, 1 negative per anchor
        Interpolates between (pos_weight=1.0, neg_weight=0.0) at step 0 and (pos_weight=0.67, neg_weight=0.33) at step total_steps
        """
        neg_proportion = 0.33 * min(1.0, self.step / self.total_steps)
        pos_weight = torch.tensor(max(0.0, 1 - neg_proportion))
        neg_weight = torch.tensor(neg_proportion)
        self.step += 1
        return pos_weight, neg_weight

class LinearLateNeg(LossSchedule):
    """Starts with only positive examples for NEG_START steps, then
    linearly increases negative weight to 0.25 by the end of training
    (doesn't quite get to the top proportion because of the late start)"""
    def __init__(self, total_steps: int):
        super().__init__(total_steps=total_steps)

    def __call__(self):
        NEG_START = 1000
        if self.step < NEG_START:
            pos_weight = torch.tensor(1.)
            neg_weight = torch.tensor(0.)
        else:
            neg_proportion = 0.25 * min(1.0, (self.step - NEG_START) / self.total_steps)
            pos_weight = torch.tensor(max(0.0, 1 - neg_proportion))
            neg_weight = torch.tensor(neg_proportion)
        self.step += 1
        return pos_weight, neg_weight


class PositiveOnlySchedule(LossSchedule):
    def __init__(self, total_steps: int):
        super().__init__(total_steps=total_steps)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(1.0), torch.tensor(0.0)
