import torch
import torch.nn as nn
from model.base import BaseFeatureExtractor

class MLPExtractor(BaseFeatureExtractor):
    def __init__(self,
                 state_dim: int,
                 feature_dim: int,
                 layer_count: int = 2
                 ) -> None:
        super().__init__(state_dim=state_dim, 
                         feature_dim=feature_dim, 
                         name='MLP')

        layers = []
        layers.append(nn.Linear(state_dim, feature_dim)) # input layer
        layers.append(nn.Mish())
        for _ in range(layer_count):
            layers.append(nn.Linear(feature_dim, feature_dim))
            layers.append(nn.Mish())

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)

if __name__ == '__main__':
    mlp = MLPExtractor(state_dim=10, feature_dim=256)
    print(mlp)
    print(mlp(torch.randn(5, 10)).shape)
