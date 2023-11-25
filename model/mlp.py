import torch
import torch.nn as nn
from torch.functional import F


class MLP(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 act_dim: int,
                 embed_dim: int = 256,
                 squash_act: bool = True
                 ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, act_dim),
        )
        
        self.squash_act = squash_act

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        act = self.mlp(feature)
        if self.squash_act:
            return F.tanh(act)
        else:
            return act
