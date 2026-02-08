"""
Rank Selector V2 Model & Loss
Two-head architecture with Weighted Pairwise Ranking Loss and BCE Trade Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class TwoHeadRankSelector(nn.Module):
    """
    Two-Head Rank Selector Model.
    Inputs: [B, N, F] normalized feature vectors.
    Outputs:
      - score: [B, N] real-valued ranking score.
      - p_trade: [B, N] probability [0, 1] for do-not-trade.
    """
    def __init__(self, 
                 input_dim: int = 5,
                 hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        # Shared Encoder
        # Simple MLP or deeper?
        # User spec didn't specify depth, assuming reasonable MLP.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Rank Head (Score)
        self.rank_head = nn.Linear(hidden_dim, 1)
        
        # Trade Head (Probability)
        self.trade_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, F]
        """
        # Shared representation
        h = self.encoder(x) # [B, N, H]
        
        # Heads
        score = self.rank_head(h).squeeze(-1) # [B, N]
        p_trade = self.trade_head(h).squeeze(-1) # [B, N]
        
        return score, p_trade

class WeightedPairwiseLoss(nn.Module):
    """
    Composite Loss:
    L = L_rank + lambda * L_trade
    
    L_rank: Boolean-Weighted Pairwise Logistic Loss.
    L_trade: Weighted BCE Loss.
    """
    def __init__(self, trade_lambda: float = 0.25, top_bottom_quantile: float = 0.20, max_pairs: int = 2000, seed: int = 42):
        super().__init__()
        self.trade_lambda = trade_lambda
        self.q = top_bottom_quantile
        self.max_pairs = max_pairs
        self.seed = seed
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, 
                scores: torch.Tensor, # [B, N]
                p_trade: torch.Tensor, # [B, N]
                y_rank: torch.Tensor, # [B, N]
                y_trade: torch.Tensor, # [B, N]
                weights: torch.Tensor, # [B, N]
                mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and metrics.
        """
        batch_size = scores.shape[0]
        total_rank_loss = 0.0
        total_trade_loss = 0.0
        valid_batches = 0
        
        # Per-day (per-sample in batch) computation for Ranking
        # Because pairs are constructed within the day.
        
        # Set seed ensures determinism across epochs if we re-seed each call?
        # Or just use a generator.
        rng = torch.Generator(device=scores.device)
        rng.manual_seed(self.seed) # Reset seed or increment? 
        # CAUTION: Resetting seed every batch makes sampling identical for same data? Yes.
        # But if data is shuffled, batch content changes. So it's fine?
        # User requested specific determinism.
        
        for b in range(batch_size):
            # Mask valid items
            mask_b = mask[b]
            if not mask_b.any():
                continue
            
            s = scores[b][mask_b]
            y = y_rank[b][mask_b]
            w = weights[b][mask_b]
            pt = p_trade[b][mask_b]
            yt = y_trade[b][mask_b]
            N = s.shape[0]
            k = max(1, int(N * self.q))
            
            # Sort by y_rank to find Top/Bottom sets
            # We want indices.
            sorted_idx = torch.argsort(y, descending=True)
            top_idx = sorted_idx[:k]
            bottom_idx = sorted_idx[-k:]
            
            # Create pairs (i, j) where i in Top, j in Bottom
            # Total possible pairs = k * k
            # Sample max_pairs
            
            # We can construct grid
            grid_i, grid_j = torch.meshgrid(top_idx, bottom_idx, indexing='ij')
            grid_i = grid_i.flatten()
            grid_j = grid_j.flatten()
            
            num_pairs = grid_i.shape[0]
            if num_pairs > self.max_pairs:
                # Random sample
                perm = torch.randperm(num_pairs, generator=rng)[:self.max_pairs]
                idx_i = grid_i[perm]
                idx_j = grid_j[perm]
            else:
                idx_i = grid_i
                idx_j = grid_j
                
            if len(idx_i) > 0:
                # Score difference: s_i - s_j
                # We want s_i > s_j (since y_i > y_j by construction top/bottom)
                score_diff = s[idx_i] - s[idx_j]
                
                # Pair weight: sqrt(w_i * w_j)
                w_pair = torch.sqrt(w[idx_i] * w[idx_j])
                
                # Logistic Loss: log(1 + exp(-diff))
                # softplus(-diff)
                loss_ij = F.softplus(-score_diff)
                
                # Weighted Mean
                l_rank = (loss_ij * w_pair).sum() / (w_pair.sum() + 1e-8)
                total_rank_loss += l_rank
            
            # --- Trade Loss ---
            # Weighted BCE
            
            # pt and yt already extracted at start of loop
            bce = self.bce(pt, yt)
            # w also extracted
            
            l_trade = (bce * w).sum() / (w.sum() + 1e-8)
            total_trade_loss += l_trade
            
            valid_batches += 1
            
        avg_rank_loss = total_rank_loss / max(1, valid_batches)
        avg_trade_loss = total_trade_loss / max(1, valid_batches)
        
        total_loss = avg_rank_loss + self.trade_lambda * avg_trade_loss
        
        return total_loss, {
            "loss_rank": avg_rank_loss, 
            "loss_trade": avg_trade_loss,
            "mean_p_trade": p_trade[mask].mean() if mask.any() else torch.tensor(0.0)
        }
