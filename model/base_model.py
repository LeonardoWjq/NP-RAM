from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFeatureExtractor(ABC, nn.Module):
    def __init__(self, state_dim: int, name: str) -> None:
        super().__init__()
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def forward(self,
                state: torch.Tensor,
                image: torch.Tensor) -> torch.Tensor:
        pass
