"""
Weighted InfoNCE loss.

  L = -sim(q, p+)/tau + log( exp(sim(q, p+)/tau) + sum_j w_j * exp(sim(q, n_j)/tau) )

When all w_j = 1.0 this is standard InfoNCE.

For numerical stability the negatives term is computed as:
  logsumexp( log(w_j) + sim(q, n_j)/tau )

which avoids overflow before the log.
"""

import torch
import torch.nn.functional as F


def weighted_infonce(
    query: torch.Tensor,        # (B, dim)  — already L2-normalized
    positive: torch.Tensor,     # (B, dim)  — fixed, already L2-normalized
    negatives: torch.Tensor,    # (B, N, dim) — fixed, already L2-normalized
    weights: torch.Tensor,      # (B, N)
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Returns mean scalar loss over the batch.
    """
    # sim(q, p+): (B,)
    sim_pos = (query * positive).sum(dim=-1) / temperature

    # sim(q, n_j): (B, N)
    sim_neg = torch.bmm(negatives, query.unsqueeze(-1)).squeeze(-1) / temperature  # (B, N)

    # log-sum-exp trick: log( sum_j w_j * exp(sim_neg_j) ) = logsumexp( log(w_j) + sim_neg_j )
    log_weights = torch.log(weights.clamp(min=1e-8))
    neg_lse = torch.logsumexp(log_weights + sim_neg, dim=-1)  # (B,)

    # Positive term in denominator
    denom = torch.logaddexp(sim_pos, neg_lse)  # (B,)

    loss = -sim_pos + denom
    return loss.mean()
