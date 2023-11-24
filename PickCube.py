import os
from pathlib import Path

import gymnasium as gym
import mani_skill2.envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from data.dataset import PickCubeDataset
from utils.data_utils import make_path

data_path = make_path('demonstrations',
                      'v0',
                      'rigid_body',
                      'PickCube-v0',
                      )

log_path = make_path('logs', 'PickCube-v0')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')

Path(log_path).mkdir(exist_ok=True, parents=True)
Path(ckpt_path).mkdir(exist_ok=True, parents=True)
Path(tb_path).mkdir(exist_ok=True, parents=True)


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


class Policy(nn.Module):
    def __init__(self,
                 in_channels: int = 8,
                 mid_channels: int = 16,
                 out_channels: int = 8,
                 obs_dim: int = 35,
                 act_dim: int = 8) -> None:
        super().__init__()
        self.resnet = ResNet(in_channels, mid_channels, out_channels)
        self.image_linear = nn.Linear(32*32*out_channels, 256)
        self.state_linear = nn.Linear(obs_dim, 64)

        self.mlp = nn.Sequential(
            nn.Linear(64 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self,
                state: torch.Tensor,
                image: torch.Tensor) -> torch.Tensor:

        state = self.state_linear(state)
        state_feature = F.relu(state)

        image = self.resnet(image)
        image = self.image_linear(image)
        image_feature = F.relu(image)

        feature = torch.cat([state_feature, image_feature], dim=1)
        out = self.mlp(feature)
        return F.tanh(out)


def train(data_path: str,
          load_count: int = 5,
          batch_size: int = 64,
          epoch: int = 10,
          lr: float = 1e-3,
          seed: int = 42,
          ckpt_freq: int = 5):

    torch.random.manual_seed(seed)
    np.random.seed(seed)

    print('Loading data...')
    dataset = PickCubeDataset(data_path, load_count=load_count)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    writer = SummaryWriter(tb_path)

    pol = Policy()
    optimizer = optim.Adam(pol.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    dummy_state = torch.zeros((1, 35))
    dummy_image = torch.zeros((1, 128, 128, 8))
    writer.add_graph(pol, (dummy_state, dummy_image))

    print('Start training...')
    mean_loss = []
    for epoch in tqdm(range(1, epoch+1)):
        total_loss = 0

        for state, image, action in dataloader:
            optimizer.zero_grad()
            pred = pol(state, image)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(state)

        if epoch % ckpt_freq == 0:
            torch.save(pol.state_dict(),
                       os.path.join(ckpt_path, f'ckpt_{epoch}.pt'))

        mean = total_loss/len(dataset)
        mean_loss.append(mean)
        writer.add_scalar('Loss', mean, epoch)

    with open(os.path.join(log_path, 'loss.json'), 'w') as f:
        json.dump(mean_loss, f, indent=4)


if __name__ == '__main__':

    train(data_path,
          load_count=998,
          batch_size=128,
          epoch=50,
          lr=1e-3,
          seed=42,
          ckpt_freq=2)

    # env = gym.make('PickCube-v0',
    #                obs_mode=['rgbd', 'state'],
    #                control_mode='pd_joint_delta_pos',
    #                reward_mode='normalized_dense',
    #                render_mode='rgb_array')
    # obs, info = env.reset()
    # agent = obs['agent']
    # extra = obs['extra']
    # camera = obs['camera_param']
    # for val in agent.values():
    #     print(val.shape)
    # for val in extra.values():
    #     print(val.shape)
    # for val in camera.values():
    #     for val2 in val.values():
    #         print(val2.shape)

    # env.close()
