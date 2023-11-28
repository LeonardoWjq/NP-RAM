import torch
import torch.nn as nn

from model.base import BaseRegressor


class LinearRegressor(BaseRegressor):
    def __init__(self,
                 feature_dim: int,
                 act_dim: int,
                 embed_dim: int = 256,
                 layer_count: int = 0
                 ) -> None:
        super().__init__(feature_dim=feature_dim, 
                         act_dim=act_dim)

        if layer_count > 0:
            layers = [nn.Linear(feature_dim, embed_dim),
                      nn.Mish()] # input layer
            for _ in range(layer_count - 1):
                layers.append(nn.Linear(embed_dim, embed_dim))
                layers.append(nn.Mish())
            layers.append(nn.Linear(embed_dim, act_dim)) # output layer
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = nn.Linear(feature_dim, act_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.mlp(feature)


if __name__ == '__main__':
    linear = LinearRegressor(feature_dim=128, 
                             act_dim=7,
                             layer_count=2)
    print(linear)
    print(linear(torch.randn(5, 128)).shape)