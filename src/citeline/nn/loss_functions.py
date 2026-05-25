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

class BasicCosineLoss(ContrastiveLossFunction):
    def __init__(self, loss_schedule=None):
        assert loss_schedule is None, "BasicCosineLoss does not support a loss schedule"
        super().__init__(loss_schedule=None)
    
    def __call__(self, anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, training: bool = True) -> torch.Tensor:
        ones = torch.ones(anchor.size(0), device=anchor.device)
        loss = F.cosine_embedding_loss(anchor, positives, ones, margin=0.0)
        return loss.mean()

class BasicTripletCosineLoss(ContrastiveLossFunction):
    def __init__(self, margin=0.1, loss_schedule=None):
        super().__init__()
        self.margin = margin

    def __call__(self, anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Assumes one positive and one negative per anchor.
        Loss = max(0, sim(anchor, negative) - sim(anchor, positive) + margin)
        """
        positives, negatives
        device = anchor.device

        if self.loss_schedule is not None and training:
            pos_weight, neg_weight = self.loss_schedule()
        else:
            pos_weight, neg_weight = torch.tensor(1.0), torch.tensor(1.0)
        pos_weight, neg_weight = pos_weight.to(device), neg_weight.to(device)

        # ones = torch.ones(anchor.size(0), device=anchor.device)
        sim_pos = F.cosine_similarity(anchor, positives)
        sim_neg = F.cosine_similarity(anchor, negatives)

        pos_loss = torch.max(0, 0.85 - sim_pos)
        neg_loss = torch.max(0, sim_neg - 0.70)
        loss = pos_weight * pos_loss + neg_weight * neg_loss
        return loss.mean()
        # loss = pos_weight * F.cosine_embedding_loss(
        #     anchor, positives, ones, margin=0.1
        # ) + neg_weight * F.cosine_embedding_loss(anchor, negatives, -ones, margin=0.1)
        # return loss.mean()
    
        # pos_sim = F.cosine_similarity(anchor, positives)
        # neg_sim = F.cosine_similarity(anchor, negatives)
        # loss = F.relu(neg_sim - pos_sim + self.margin)
        # return loss.mean()