from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class ContrastiveLossFunction(ABC):
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ContrastiveLossFunction.registry[cls.__name__] = cls

    @abstractmethod
    def __call__(self, anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor: ...


class BasicTripletCosineLoss(ContrastiveLossFunction):
    def __call__(self, anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """Assumes one positive and one negative per anchor.
        Loss = max(0, sim(anchor, negative) - sim(anchor, positive) + margin)
        """
        positive, negative = positives, negatives
        ones = torch.ones(anchor.size(0), device=anchor.device)
        loss = F.cosine_embedding_loss(
            anchor, positive, ones, margin=0.1
        ) + F.cosine_embedding_loss(anchor, negative, -ones, margin=0.1)
        return loss.mean()
