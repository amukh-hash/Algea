import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherLoss(nn.Module):
    """
    Teacher Loss: Cross-Entropy only.
    We want the Teacher (T5) to learn a calibrated probability distribution
    over the token bins, without any risk-weighting bias.
    """
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, teacher_logits, target_token_ids):
        """
        teacher_logits: (Batch, Time, Vocab)
        target_token_ids: (Batch, Time)
        """
        logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        targets_flat = target_token_ids.view(-1)
        return self.ce_loss(logits_flat, targets_flat)


def distillation_loss(student_logits, teacher_logits, true_labels, T=2.0, alpha=0.5):
    """
    Knowledge Distillation loss combining Hard and Soft targets.
    
    Args:
        student_logits: Unnormalized logits from Student
        teacher_logits: Unnormalized logits from Teacher
        true_labels: Ground truth labels (0, 1, 2)
        T: Temperature for softening distributions
        alpha: Weight for Hard Loss (1.0=pure hard, 0.0=pure soft)
    
    Returns:
        Combined loss scalar
    """
    # 1. Hard Loss (Student vs Ground Truth)
    hard_loss = F.cross_entropy(student_logits, true_labels)

    # 2. Soft Loss (Student vs Teacher)
    # KLDivLoss requires: Input=LogSoftmax, Target=Softmax
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    
    student_log_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    
    # T^2 scaling keeps gradients balanced with hard_loss
    soft_loss = kl_criterion(student_log_soft, teacher_soft) * (T * T)

    # 3. Combine
    return (alpha * hard_loss) + ((1 - alpha) * soft_loss)

def teacher_logits_to_quantiles(logits, bin_centers, quantiles=[0.1, 0.5, 0.9]):
    """
    Convert T5 logits (probability distribution) to quantile values.
    Used to generate 'ground truth' quantile targets for the Student (Bolt).
    
    logits: (Batch, Time, Vocab)
    bin_centers: (Vocab,) Tensor of values for each bin
    quantiles: List of quantiles to extract (must match Student's quantiles)
    """
    probs = F.softmax(logits, dim=-1)  # (B, H, Vocab)
    cdf = torch.cumsum(probs, dim=-1)
    
    # Ensure centers are on the same device
    if bin_centers.device != logits.device:
        bin_centers = bin_centers.to(logits.device)
        
    results = []
    # Vectorized search would be faster but for now loop is clear
    # We find the first index where CDF >= q
    for q in quantiles:
        # (cdf >= q).long() gives 1s where true. argmax gives first index of max.
        # But if none are true (e.g. q=1.0 and float errors), it picks 0?
        # CDF ends at 1.0, so >=q should occur.
        idx = (cdf >= q).float().argmax(dim=-1)
        values = bin_centers[idx]
        results.append(values)
        
    return torch.stack(results, dim=1)  # (B, Num_Q, H)

class StudentMultiObjectiveLoss(nn.Module):
    """
    Student Loss: Multi-Objective Optimization for Chronos Bolt.
    1. Distillation: Match Teacher's projected quantiles (Huber Loss)
    2. Sortino: Maximize risk-adjusted return (on predicted median)
    3. Direction: Match TBM binary label (Sign alignment)
    4. Feature: Align hidden states (MSE)
    
    Weighted automatically using Homoscedastic Uncertainty.
    """
    def __init__(self, risk_free_rate=0.0, quantiles=[0.1, 0.5, 0.9], 
                 student_dim=512, teacher_dim=768): # Default dimensions, verify actuals
        super().__init__()
        self.rfr = risk_free_rate
        # Broadcasting shape: (1, 3, 1) if 3 quantiles
        self.register_buffer('quantiles_tensor', torch.tensor(quantiles).view(1, -1, 1))
        
        # Learnable weights for 4 tasks: [Distill, Sortino, Direction, Feature]
        self.log_vars = nn.Parameter(torch.zeros(4))
        
        # Projector for Feature Alignment (Student -> Teacher dim)
        self.distill_projector = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_quantiles, teacher_quantiles, tbm_labels, 
                student_hidden=None, teacher_hidden=None):
        """
        student_quantiles: (Batch, Num_Quantiles, Horizon) - Raw outputs from Bolt
        teacher_quantiles: (Batch, Num_Quantiles, Horizon) - Target values from T5
        tbm_labels: (Batch,) - Triple Barrier labels (1=Buy, 0=Neutral/Sell)
        student_hidden: (Batch, Time, Dim) - Optional for feature loss
        teacher_hidden: (Batch, Time, Dim)
        """
        
        # --- 1. Distillation (Quantile Regression) ---
        # Huber Loss is robust to outliers
        loss_distill = F.smooth_l1_loss(student_quantiles, teacher_quantiles.detach())
        
        # --- 2. Differentiable Sortino ---
        # Use the Median (index 1 for [0.1, 0.5, 0.9]) as the "Expected Price"
        # We assume the user config is always [0.1, 0.5, 0.9] or similar where middle is median
        median_idx = student_quantiles.shape[1] // 2 
        median_forecast = student_quantiles[:, median_idx, :] 
        
        # Calculate Returns: (P_t / P_t-1) - 1
        returns = torch.diff(median_forecast, dim=1) / (median_forecast[:, :-1] + 1e-8)
        
        # Downside Deviation (Risk)
        downside = torch.clamp(returns - self.rfr, max=0)
        downside_std = torch.sqrt(torch.mean(downside**2, dim=1) + 1e-6)
        mean_return = torch.mean(returns, dim=1)
        
        # SortinoRatio = Mean / DownsideDev
        # Minimize Negative Sortino
        sortino = mean_return / (downside_std + 1e-6)
        loss_sortino = -torch.mean(sortino)
        
        # --- 3. Directional Alignment (Proxy for Focal) ---
        # Did we predict the correct direction?
        # Proxy: Sigmoid of mean return vs TBM Label
        pred_prob = torch.sigmoid(mean_return) # 0..1 based on return magnitude
        
        # Ensure targets match shape (Batch,)
        if tbm_labels.ndim > 1:
            targets = tbm_labels.float().mean(dim=1) # Aggregate if needed
        else:
            targets = tbm_labels.float()
            
        # Binary Cross Entropy
        bce = F.binary_cross_entropy(pred_prob, targets, reduction='none')
        # Focal Weighting: (1-pt)^2 * BCE
        loss_direction = ((1 - torch.exp(-bce))**2.0 * bce).mean()
        
        # --- 4. Feature Projection (Optional) ---
        if student_hidden is not None and teacher_hidden is not None:
            # Project Student -> Teacher space
            student_proj = self.distill_projector(student_hidden)
            loss_feature = F.mse_loss(student_proj, teacher_hidden.detach())
        else:
            loss_feature = torch.tensor(0.0, device=student_quantiles.device)

        # --- Weighted Sum (Homoscedastic Uncertainty) ---
        sigma = torch.exp(self.log_vars)
        
        l1 = (loss_distill / (2 * sigma[0])) + 0.5 * self.log_vars[0]
        l2 = (loss_sortino / (2 * sigma[1])) + 0.5 * self.log_vars[1]
        l3 = (loss_direction / (2 * sigma[2])) + 0.5 * self.log_vars[2]
        l4 = (loss_feature / (2 * sigma[3])) + 0.5 * self.log_vars[3]
        
        total_loss = l1 + l2 + l3 + l4
        
        return total_loss, (loss_distill.item(), loss_sortino.item(), loss_direction.item(), loss_feature.item())
