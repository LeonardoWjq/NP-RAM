import gymnasium as gym
import mani_skill2.envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PickCubeDataset


class ResBlock(nn.Module):
    def __init__(self,
                 channel: int,
                 kernel_size: int,
                 stride: int,
                 ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel,
                               kernel_size, stride=stride, padding='same')
        self.conv2 = nn.Conv2d(channel, channel,
                               kernel_size, stride=stride, padding='same')
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.conv1(x)
        feature = F.relu(feature)
        feature = self.bn(feature)
        feature = self.conv2(x)
        return x + feature


class ResNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 kernel_size: int = 3,
                 stride: int = 1,
                 ) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, mid_channels,
                                 kernel_size=kernel_size, stride=stride,
                                 padding='same')

        self.in_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        blocks = []

        for _ in range(num_blocks):
            blocks.append(ResBlock(mid_channels, kernel_size, stride))
            blocks.append(nn.ReLU())
            blocks.append(nn.BatchNorm2d(mid_channels))

        self.res_blocks = nn.Sequential(*blocks)

        self.out_conv = nn.Conv2d(mid_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding='same')

        self.out_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)

        x = self.in_conv(x)
        x = F.relu(x)
        x = self.in_pooling(x)

        x = self.res_blocks(x)

        x = self.out_conv(x)
        x = F.relu(x)
        x = self.out_pooling(x)

        return torch.flatten(x, start_dim=1)


if __name__ == '__main__':
    env = gym.make('PickCube-v0',
                   obs_mode=['rgbd', 'state'],
                   control_mode='pd_joint_delta_pos',
                   reward_mode='normalized_dense',
                   render_mode='rgb_array')
    obs, info = env.reset()
    agent = obs['agent']
    extra = obs['extra']
    camera = obs['camera_param']
    for val in agent.values():
        print(val.shape)
    for val in extra.values():
        print(val.shape)
    for val in camera.values():
        for val2 in val.values():
            print(val2.shape)
        
    
    env.close()
