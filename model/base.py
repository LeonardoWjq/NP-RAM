from abc import ABC, abstractmethod
from typing import Tuple, Union
import torch
import torch.nn as nn


class BaseFeatureExtractor(ABC, nn.Module):
    def __init__(self,
                 state_dim: int,
                 feature_dim: int,
                 name: str) -> None:
        super().__init__()
        self.name = name
        self.feature_dim = feature_dim

    def get_name(self):
        return self.name
    
    def get_feature_dim(self) -> int:
        return self.feature_dim

    @abstractmethod
    def forward(self,
                obs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        pass


class BaseRegressor(ABC, nn.Module):
    def __init__(self,
                 feature_dim: int,
                 act_dim: int) -> None:
        super().__init__()

    def get_name(self):
        return self.name

    @abstractmethod
    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        pass


class BasePolicy(nn.Module):
    def __init__(self,
                 feature_extractor: BaseFeatureExtractor,
                 regressor: BaseRegressor,
                 squash_output: bool = True) -> None:
        super().__init__()
        self._feature_extractor = feature_extractor
        self._regressor = regressor
        self._squash_output = squash_output
    
    def forward(self,
                obs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        feature = self._feature_extractor(obs)
        action = self._regressor(feature)
        if self._squash_output:
            action = torch.tanh(action)
        return action
    
    def get_feature_extractor(self) -> BaseFeatureExtractor:
        return self._feature_extractor
    
    def get_regressor(self) -> BaseRegressor:
        return self._regressor