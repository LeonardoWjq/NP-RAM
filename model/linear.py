import torch
import torch.nn as nn

from model.base import BaseRegressor


class LinearRegressor(BaseRegressor):
    def __init__(self,
                 feature_dim: int,
                 act_dim: int,
                 layer_count: int = 0
                 ) -> None:
        super().__init__(feature_dim=feature_dim, 
                         act_dim=act_dim)

        layers = []
        for _ in range(layer_count):
            layers.append(nn.Linear(feature_dim, feature_dim))
            layers.append(nn.Mish())
        layers.append(nn.Linear(feature_dim, act_dim)) # output layer

        self.mlp = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.mlp(feature)


if __name__ == '__main__':
    linear = LinearRegressor(feature_dim=256, act_dim=7)
    print(linear)
    print(linear(torch.randn(5, 256)).shape)