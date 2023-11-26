import json
import os
from pathlib import Path
from typing import List

import gymnasium as gym
import mani_skill2.envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mani_skill2.utils.wrappers import RecordEpisode
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import ManiskillStateDataset
from model.resnet import ResNetFeatureExtractor
from model.mlp import MLP
from utils.data_utils import (flatten_state, make_path, process_image,
                              rescale_rgbd)

ENV_ID = 'LiftCube-v0'
OBS_MODE = 'state'
CONTROL_MODE = 'pd_ee_delta_pose'

data_path = make_path('demonstrations',
                      'v0',
                      'rigid_body',
                      ENV_ID,
                      f'trajectory.{OBS_MODE}.{CONTROL_MODE}.h5'
                      )

log_path = make_path('logs', f'BC-{ENV_ID}-{OBS_MODE}-{CONTROL_MODE}')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')
video_path = os.path.join(log_path, 'videos')

Path(log_path).mkdir(exist_ok=True, parents=True)
Path(ckpt_path).mkdir(exist_ok=True, parents=True)
Path(tb_path).mkdir(exist_ok=True, parents=True)
Path(video_path).mkdir(exist_ok=True, parents=True)

gpu_id = 'cuda:1'
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')


class MLPExtractor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 embed_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
            nn.Mish()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class Policy(nn.Module):
    def __init__(self, feature_extractor: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mlp = mlp

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feature = self.feature_extractor(state)
        return self.mlp(feature)


def train(data_path: str,
          state_dim: int,
          act_dim: int,
          feature_dim: int = 512,
          load_count: int = None,
          batch_size: int = 64,
          epoch: int = 10,
          lr: float = 1e-3,
          weight_decay: float = 1e-5,
          seed: int = 42,
          ckpt_freq: int = 5,
          start_epoch: int = 0) -> str:

    torch.random.manual_seed(seed)
    np.random.seed(seed)

    print('Loading data...')
    dataset = ManiskillStateDataset(data_path, load_count=load_count)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)

    writer = SummaryWriter(tb_path)

    if start_epoch > 0:
        print('Loading checkpoint...')

        ckpt = os.path.join(ckpt_path, f'ckpt_{start_epoch}.pt')
        ckpt = torch.load(ckpt)
        feature_extractor = MLPExtractor(state_dim=state_dim,
                                         embed_dim=feature_dim)
        feature_extractor.load_state_dict(ckpt['feature_extractor_state_dict'])

        mlp = MLP(feature_dim=feature_dim,
                  act_dim=act_dim)
        mlp.load_state_dict(ckpt['mlp_state_dict'])

        policy = Policy(feature_extractor, mlp)
        policy.to(device)
        policy.train()

        optimizer = optim.RAdam(policy.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        with open(os.path.join(log_path, 'train_log.json'), 'r') as f:
            log = json.load(f)
            mean_loss = log['mean_loss']
            min_loss = log['min_loss']
            best_ckpt = log['best_ckpt']
    else:
        print('Starting from scratch...')
        feature_extractor = MLPExtractor(state_dim=state_dim,
                                         embed_dim=feature_dim)

        mlp = MLP(feature_dim=feature_dim,
                  act_dim=act_dim)

        policy = Policy(feature_extractor, mlp)
        policy.to(device)
        policy.train()

        optimizer = optim.RAdam(policy.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)

        dummy_state = torch.zeros((1, state_dim)).to(device)
        writer.add_graph(policy, dummy_state)

        mean_loss = []
        min_loss = np.inf
        best_ckpt = None

    criterion = nn.MSELoss(reduction='mean')

    print('Start training...')

    for epoch in tqdm(range(start_epoch+1, epoch+1)):
        total_loss = 0

        for state, action in dataloader:
            state = state.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            pred = policy(state)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(state)

        if epoch % ckpt_freq == 0:
            ckpt = os.path.join(ckpt_path, f'ckpt_{epoch}.pt')
            torch.save(dict(feature_extractor_state_dict=feature_extractor.state_dict(),
                            mlp_state_dict=mlp.state_dict(),
                            optimizer_state_dict=optimizer.state_dict()),
                       ckpt)
            if total_loss < min_loss:
                min_loss = total_loss
                best_ckpt = ckpt

        mean = total_loss/len(dataset)
        mean_loss.append(mean)
        writer.add_scalar('Loss', mean, epoch)

    log = dict(mean_loss=mean_loss,
               best_ckpt=best_ckpt,
               min_loss=min_loss)

    with open(os.path.join(log_path, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=4)

    writer.flush()
    writer.close()

    return best_ckpt


def test(ckpt: str,
         state_dim: int,
         act_dim: int,
         feature_dim: int = 512,
         max_steps: int = 250,
         num_episodes: int = 100) -> List[int]:

    env = gym.make(id=ENV_ID,
                   obs_mode=OBS_MODE,
                   control_mode=CONTROL_MODE,
                   max_episode_steps=max_steps,
                   renderer_kwargs={'device': gpu_id})

    feature_extractor = MLPExtractor(state_dim=state_dim,
                                     embed_dim=feature_dim)

    mlp = MLP(feature_dim=feature_dim,
              act_dim=act_dim)

    ckpt = torch.load(ckpt)
    feature_extractor.load_state_dict(ckpt['feature_extractor_state_dict'])
    mlp.load_state_dict(ckpt['mlp_state_dict'])
    policy = Policy(feature_extractor, mlp)
    policy.to(device)
    policy.eval()

    success_seeds = []
    returns = {}

    with torch.no_grad():
        for seed in tqdm(range(num_episodes)):
            obs, _ = env.reset(seed=seed)
            G = 0
            terminated = False
            truncated = False
            while not terminated and not truncated:
                state = torch.from_numpy(obs)
                state = state.unsqueeze(0).to(device)

                action = policy(state)
                action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action[0])
                G += reward

            if info['success']:
                success_seeds.append(seed)

            returns[seed] = G

    env.close()

    log = dict(returns=returns,
               max_steps=max_steps,
               num_episodes=num_episodes,
               success_rate=len(success_seeds) / num_episodes,
               success_seeds=success_seeds)

    with open(os.path.join(log_path, 'test_log.json'), 'w') as f:
        json.dump(log, f, indent=4)

    return success_seeds


def render_video(ckpt: str,
                 state_dim: int,
                 act_dim: int,
                 seed: int,
                 feature_dim: int = 512,
                 max_steps: int = 250) -> None:

    env = gym.make(id=ENV_ID,
                   render_mode="rgb_array",
                   enable_shadow=True,
                   obs_mode=OBS_MODE,
                   control_mode=CONTROL_MODE,
                   max_episode_steps=max_steps,
                   renderer_kwargs={'device': gpu_id})

    env = RecordEpisode(
        env,
        video_path,
        info_on_video=True,
        save_trajectory=False
    )

    feature_extractor = MLPExtractor(state_dim=state_dim,
                                     embed_dim=feature_dim)
    
    mlp = MLP(feature_dim=feature_dim,
              act_dim=act_dim)

    ckpt = torch.load(ckpt)
    feature_extractor.load_state_dict(ckpt['feature_extractor_state_dict'])
    mlp.load_state_dict(ckpt['mlp_state_dict'])
    policy = Policy(feature_extractor, mlp)
    policy.to(device)
    policy.eval()

    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False

    with torch.no_grad():
        while not terminated and not truncated:
            state = torch.from_numpy(obs)
            state = state.unsqueeze(0).to(device)
            action = policy(state)
            action = action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action[0])

    env.flush_video(suffix=f'{ENV_ID}_{OBS_MODE}_{CONTROL_MODE}_{seed}')
    env.close()


if __name__ == '__main__':
    STATE_DIM = 42
    ACT_DIM = 7
    MAX_STEPS = 300
    NUM_EPISODES = 100

    ckpt = train(data_path,
                 state_dim=STATE_DIM,
                 act_dim=ACT_DIM,
                 load_count=None,
                 batch_size=128,
                 epoch=300,
                 lr=1e-3,
                 seed=42,
                 ckpt_freq=2,
                 start_epoch=0)

    # ckpt = os.path.join(ckpt_path, 'ckpt_82.pt')

    success_seeds = test(ckpt,
                         state_dim=STATE_DIM,
                         act_dim=ACT_DIM,
                         max_steps=MAX_STEPS,
                         num_episodes=NUM_EPISODES)
    # success_seeds = [54]
    for seed in success_seeds[:5]:
        render_video(ckpt,
                     state_dim=STATE_DIM,
                     act_dim=ACT_DIM,
                     seed=seed,
                     max_steps=MAX_STEPS)
