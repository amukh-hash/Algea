import torch
import torch.nn as nn
import torch.nn.functional as F

def listwise_softmax_loss(scores: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Listwise ranking loss using softmax cross-entropy.
    scores: [N, 1] predicted scores
    targets: [N, 1] true relevance (e.g., returns)
    """
    # Normalize targets to probability distribution
    # If targets are returns, can be negative.
    # Convert to ranking probability: p_i = exp(target_i / T) / sum(exp(target_j / T))
    # Or simply: softmax(targets)

    # Ensure shapes
    scores = scores.view(-1)
    targets = targets.view(-1)

    # Softmax on scores
    pred_dist = F.softmax(scores / temperature, dim=0)

    # Softmax on targets (ground truth distribution)
    # Using a higher temperature for targets can smoothen the labels
    true_dist = F.softmax(targets / temperature, dim=0)

    # KL Divergence: sum(true * log(true / pred))
    # PyTorch KLDivLoss expects input as log-probabilities
    loss = F.kl_div(pred_dist.log(), true_dist, reduction='batchmean')

    return loss

def pairwise_margin_loss(scores: torch.Tensor, targets: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """
    Pairwise ranking loss (MarginRankingLoss).
    Sample pairs (i, j) where target_i > target_j.
    Maximize score_i - score_j > margin.
    """
    scores = scores.view(-1)
    targets = targets.view(-1)

    # Create all pairs
    # N x N
    diff_targets = targets.unsqueeze(0) - targets.unsqueeze(1) # T_j - T_i
    diff_scores = scores.unsqueeze(0) - scores.unsqueeze(1)   # S_j - S_i

    # Mask: only pairs where i > j (target difference > 0)
    # 1 where T_i < T_j (row < col) -> T_col - T_row > 0
    # Actually let's use standard pytorch MarginRankingLoss logic
    # But for N items, easier to compute full matrix

    # Pairs where T_i > T_j
    # T_i - T_j > 0
    # We want S_i - S_j > margin

    # Matrix: M[i, j] = 1 if T_i > T_j else -1
    # Ignore ties?

    # Efficient implementation:
    # Just take random pairs or all pairs?
    # All pairs is O(N^2), feasible for N ~ 500.

    N = scores.size(0)
    if N < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # Expand
    s_i = scores.unsqueeze(1) # [N, 1]
    s_j = scores.unsqueeze(0) # [1, N]

    t_i = targets.unsqueeze(1)
    t_j = targets.unsqueeze(0)

    # Indicator: 1 if t_i > t_j, -1 if t_i < t_j, 0 if equal
    y = torch.sign(t_i - t_j)

    # Loss: max(0, -y * (s_i - s_j) + margin)
    # Only compute for y != 0
    mask = (y != 0)

    loss_mat = F.relu( -y * (s_i - s_j) + margin )

    return (loss_mat * mask).sum() / (mask.sum() + 1e-6)
