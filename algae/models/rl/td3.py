import torch
import torch.nn as nn

class TD3Actor(nn.Module):
    """
    TD3 Execution Agent Actor.
    Takes dense output embeddings from Spatial-Temporal Transformer directly into state space S_t.
    Outputs continuous action space natively dictating spread width and delta targets via reward maximization.
    """
    def __init__(self, state_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Continuous bounded actions for spread_width scale, target_delta scale
        )
        
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        state_embedding: [Batch, 256] from SpatialTemporalTransformer
        """
        return self.net(state_embedding)


class TD3Critic(nn.Module):
    """Twin Delayed DDPG (TD3) Critic Network"""
    def __init__(self, state_dim=256, action_dim=2):
        super().__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)
