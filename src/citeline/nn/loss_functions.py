from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class ContrastiveLossFunction(ABC):
    registry = {}

    def __init__(self, loss_schedule=None):
        self.loss_schedule = loss_schedule

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ContrastiveLossFunction.registry[cls.__name__] = cls

    @abstractmethod
    def __call__(self, anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, training: bool = True) -> torch.Tensor: ...


class BasicTripletCosineLoss(ContrastiveLossFunction):
    def __call__(self, anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Assumes one positive and one negative per anchor.
        Loss = max(0, sim(anchor, negative) - sim(anchor, positive) + margin)
        """
        positive, negative = positives, negatives
        device = anchor.device

        if self.loss_schedule is not None and training:
            pos_weight, neg_weight = self.loss_schedule()
        else:
            pos_weight, neg_weight = torch.tensor(1.0), torch.tensor(1.0)
        pos_weight, neg_weight = pos_weight.to(device), neg_weight.to(device)

        ones = torch.ones(anchor.size(0), device=anchor.device)
        loss = pos_weight * F.cosine_embedding_loss(
            anchor, positive, ones, margin=0.1
        ) + neg_weight * F.cosine_embedding_loss(anchor, negative, -ones, margin=0.1)
        return loss.mean()
