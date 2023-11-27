import json
import os
from pathlib import Path
from typing import List, Tuple

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

from data.dataset import ManiskillSingleDataset
from model.mlp import MLPExtractor
from model.linear import LinearRegressor
from model.base import BasePolicy
from utils.data_utils import make_path

env_id = 'LiftCube-v0'
obs_mode = 'state'
control_mode = 'pd_ee_delta_pose'


log_path = make_path('logs', f'BC-MLP-{env_id}-{obs_mode}-{control_mode}')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')
video_path = os.path.join(log_path, 'videos')

Path(log_path).mkdir(exist_ok=True, parents=True)
Path(ckpt_path).mkdir(exist_ok=True, parents=True)
Path(tb_path).mkdir(exist_ok=True, parents=True)
Path(video_path).mkdir(exist_ok=True, parents=True)

gpu_id = 'cuda:1'
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')


def validate(policy: BasePolicy,
             dataloader: DataLoader,
             criterion: nn.MSELoss) -> float:
    total_loss = 0
    policy.eval()
    with torch.no_grad():
        for state, action in dataloader:
            state = state.to(device)
            action = action.to(device)

            pred = policy(state)
            loss = criterion(pred, action)
            total_loss += loss.item()*len(state)
    policy.train()
    return total_loss


def init_training(start_epoch: int,
                  state_dim: int,
                  act_dim: int,
                  feature_dim: int,
                  lr: float,
                  weight_decay: float) -> Tuple[BasePolicy, optim.Optimizer, List[float], List[int], float, str]:

    feature_extractor = MLPExtractor(state_dim=state_dim,
                                     feature_dim=feature_dim,
                                     layer_count=2)
    regressor = LinearRegressor(feature_dim=feature_dim,
                                act_dim=act_dim)
    policy = BasePolicy(feature_extractor=feature_extractor,
                        regressor=regressor,
                        squash_output=True)
    if start_epoch > 0:
        print('Loading checkpoint...')
        ckpt = os.path.join(ckpt_path, f'ckpt_{start_epoch}.pt')
        policy.load_state_dict(ckpt['policy_state_dict'])
        optimizer = optim.RAdam(policy.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        with open(os.path.join(log_path, 'train_log.json'), 'r') as f:
            log = json.load(f)
            val_loss_list = log['val_loss_list']
            val_epochs = log['val_epochs']
            min_loss = log['min_loss']
            best_ckpt = log['best_ckpt']
    else:
        print('Starting from scratch...')
        optimizer = optim.RAdam(policy.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
        val_loss_list = []
        val_epochs = []
        min_loss = np.inf
        best_ckpt = None

    return policy, optimizer, val_loss_list, val_epochs, min_loss, best_ckpt


def save_ckpt(epoch: int,
              policy: BasePolicy,
              optimizer: optim.Optimizer) -> str:
    '''
    save the policy and optimizer state dict
    returns the path to the checkpoint
    '''
    ckpt = os.path.join(ckpt_path, f'ckpt_{epoch}.pt')
    torch.save(dict(policy_state_dict=policy.state_dict(),
                    optimizer_state_dict=optimizer.state_dict()),
               ckpt)
    return ckpt


def train(feature_dim: int = 256,
          batch_size: int = 64,
          epoch: int = 10,
          lr: float = 1e-3,
          weight_decay: float = 1e-5,
          seed: int = 42,
          ckpt_freq: int = 5,
          start_epoch: int = 0) -> str:
    '''
    Train a policy using behavioral cloning.
    Use MLP feature extractor and a linear regressor.
    Output the best checkpoint.
    '''

    print(f'Setting random seed to {seed}')
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    print('Loading data...')
    train_set = ManiskillSingleDataset(env_id,
                                       obs_mode,
                                       control_mode,
                                       train=True)
    val_set = ManiskillSingleDataset(env_id,
                                     obs_mode,
                                     control_mode,
                                     train=False)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True)

    writer = SummaryWriter(tb_path)

    state_dim = train_set.state_dim
    act_dim = train_set.act_dim

    policy, optimizer, val_loss_list, val_epochs, min_loss, best_ckpt = init_training(start_epoch,
                                                                                      state_dim,
                                                                                      act_dim,
                                                                                      feature_dim,
                                                                                      lr,
                                                                                      weight_decay)
    policy.to(device)
    policy.train()
    dummy_state = torch.zeros((1, state_dim)).to(device)
    writer.add_graph(policy, dummy_state)

    criterion = nn.MSELoss(reduction='mean')

    print('Start training...')
    for epoch in tqdm(range(start_epoch+1, epoch+1)):
        for state, action in train_loader:
            state = state.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            pred = policy(state)
            train_loss = criterion(pred, action)
            train_loss.backward()
            optimizer.step()

        writer.add_scalar('Loss/Train', train_loss.item(), epoch)

        if epoch % ckpt_freq == 0:
            ckpt = save_ckpt(epoch, policy, optimizer)
            val_loss = validate(policy, val_loader, criterion)
            val_loss_list.append(val_loss)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            if val_loss < min_loss:
                min_loss = val_loss
                best_ckpt = ckpt

    log = dict(val_loss_list=val_loss_list,
               val_epochs=val_epochs,
               best_ckpt=best_ckpt,
               min_loss=min_loss)

    with open(os.path.join(log_path, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=4)

    writer.flush()
    writer.close()

    return best_ckpt


def test(ckpt: str,
         feature_dim: int = 256,
         max_steps: int = 200,
         num_episodes: int = 30) -> List[int]:

    env = gym.make(id=env_id,
                   obs_mode=obs_mode,
                   control_mode=control_mode,
                   max_episode_steps=max_steps,
                   render_mode="rgb_array",
                   enable_shadow=True,
                   renderer_kwargs={'device': gpu_id})

    env = RecordEpisode(
        env,
        video_path,
        info_on_video=True,
        save_trajectory=True
    )

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    feature_extractor = MLPExtractor(state_dim=state_dim,
                                     feature_dim=feature_dim,
                                     layer_count=2)
    regressor = LinearRegressor(feature_dim=feature_dim,
                                act_dim=act_dim)
    policy = BasePolicy(feature_extractor=feature_extractor,
                        regressor=regressor,
                        squash_output=True)

    ckpt = torch.load(ckpt)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.to(device)
    policy.eval()

    success_seeds = []
    returns = {}

    print('Start testing...')
    with torch.no_grad():
        for seed in tqdm(range(num_episodes)):
            obs, _ = env.reset(seed=seed)
            G = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                obs = torch.from_numpy(obs)
                obs = obs.unsqueeze(0).to(device)
                action = policy(obs)
                action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action[0])
                G += reward

            if info['success']:
                success_seeds.append(seed)

            returns[seed] = G
    env.close()

    success_rate = len(success_seeds) / num_episodes
    print('Success Rate:', success_rate)
    log = dict(returns=returns,
               max_steps=max_steps,
               num_episodes=num_episodes,
               success_rate=success_rate,
               success_seeds=success_seeds)

    with open(os.path.join(log_path, 'test_log.json'), 'w') as f:
        json.dump(log, f, indent=4)

    return success_seeds


def render_video(ckpt: str,
                 seeds: List[int],
                 feature_dim: int = 256,
                 max_steps: int = 200) -> None:

    env = gym.make(id=env_id,
                   render_mode="rgb_array",
                   enable_shadow=True,
                   obs_mode=obs_mode,
                   control_mode=control_mode,
                   max_episode_steps=max_steps,
                   renderer_kwargs={'device': gpu_id})

    env = RecordEpisode(
        env,
        video_path,
        info_on_video=True,
        save_trajectory=False
    )
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    feature_extractor = MLPExtractor(state_dim=state_dim,
                                     feature_dim=feature_dim,
                                     layer_count=2)

    regressor = LinearRegressor(feature_dim=feature_dim,
                                act_dim=act_dim)
    
    policy = BasePolicy(feature_extractor=feature_extractor,
                        regressor=regressor,
                        squash_output=True)

    ckpt = torch.load(ckpt)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.to(device)
    policy.eval()

    with torch.no_grad():
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            terminated = False
            truncated = False

            while not terminated and not truncated:
                obs = torch.from_numpy(obs)
                obs = obs.unsqueeze(0).to(device)
                action = policy(obs)
                action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action[0])
    env.close()


if __name__ == '__main__':


    ckpt = train(lr=1e-3,
                 epoch=300,
                 ckpt_freq=2,
                 start_epoch=0)

    # ckpt = os.path.join(ckpt_path, 'ckpt_82.pt')

    success_seeds = test(ckpt)
