import torch
import torch.nn as nn
from torch.nn import functional as F
from model.base import BaseFeatureExtractor
from typing import Tuple

class ResBlock(nn.Module):
    def __init__(self,
                 channel: int,
                 kernel_size: int,
                 ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel,
                               kernel_size, padding='same')
        self.conv2 = nn.Conv2d(channel, channel,
                               kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.conv1(x)
        feature = F.mish(feature)
        feature = self.bn(feature)
        feature = self.conv2(x)
        return x + feature


class ResNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 kernel_size: int = 5
                 ) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, 
                                 mid_channels,
                                 kernel_size=kernel_size,
                                 padding='same')

        self.in_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        blocks = []

        for _ in range(num_blocks):
            blocks.append(ResBlock(mid_channels, kernel_size))
            blocks.append(nn.Mish())
            blocks.append(nn.BatchNorm2d(mid_channels))

        self.res_blocks = nn.Sequential(*blocks)

        self.out_conv = nn.Conv2d(mid_channels, 
                                  out_channels,
                                  kernel_size=kernel_size,
                                  padding='same')

        self.out_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)

        x = self.in_conv(x)
        x = F.mish(x)
        x = self.in_pooling(x)

        x = self.res_blocks(x)

        x = self.out_conv(x)
        x = F.mish(x)
        x = self.out_pooling(x)

        return torch.flatten(x, start_dim=1)


class ResNetExtractor(BaseFeatureExtractor):
    def __init__(self,
                 state_dim: int,
                 in_channels: int = 8,
                 mid_channels: int = 16,
                 out_channels: int = 8,
                 image_embedding: int = 256,
                 state_embedding: int = 64
                 ) -> None:
        super().__init__(state_dim=state_dim,
                         feature_dim=image_embedding + state_embedding,
                         name='ResNet')
        self._resnet = ResNet(in_channels, mid_channels, out_channels)
        self._image_linear = nn.Linear(32*32*out_channels, image_embedding)
        self._state_linear = nn.Linear(state_dim, state_embedding)

    def forward(self, state: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:
        state = self._state_linear(state)
        state_feature = F.mish(state)

        rgbd = self._resnet(rgbd)
        rgbd = self._image_linear(rgbd)
        rgbd_feature = F.mish(rgbd)

        feature = torch.cat([state_feature, rgbd_feature], dim=1)
        return feature

if __name__ == '__main__':
    resnet = ResNetExtractor(state_dim=42)
    print(resnet)
    print(resnet(torch.randn(5, 42), torch.randn(5, 128, 128, 8)).shape)