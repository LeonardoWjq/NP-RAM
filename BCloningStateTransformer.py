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

from data.dataset import ManiskillSeqDataset
from model.transformer import TransformerExtractor
from model.linear import LinearRegressor
from model.base import BasePolicy
from utils.data_utils import make_path
from utils.train_utils import init_deque, update_deque

env_id = 'LiftCube-v0'
obs_mode = 'state'
control_mode = 'pd_ee_delta_pose'


log_path = make_path(
    'logs', f'BC-Transformer-{env_id}-{obs_mode}-{control_mode}')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')
video_path = os.path.join(log_path, 'videos')

Path(log_path).mkdir(exist_ok=True, parents=True)
Path(ckpt_path).mkdir(exist_ok=True, parents=True)
Path(tb_path).mkdir(exist_ok=True, parents=True)
Path(video_path).mkdir(exist_ok=True, parents=True)

gpu_id = 'cuda:3'
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
                  seq_len: int,
                  lr: float,
                  weight_decay: float) -> Tuple[BasePolicy, optim.Optimizer, float, str]:

    feature_extractor = TransformerExtractor(state_dim=state_dim,
                                             feature_dim=feature_dim,
                                             seq_len=seq_len,
                                             layer_count=2)

    regressor = LinearRegressor(feature_dim=feature_dim,
                                act_dim=act_dim,
                                layer_count=3)

    policy = BasePolicy(feature_extractor=feature_extractor,
                        regressor=regressor,
                        squash_output=True)

    policy.to(device)

    if start_epoch > 0:
        print('Loading checkpoint...')
        ckpt = os.path.join(ckpt_path, f'ckpt_{start_epoch}.pt')
        ckpt = torch.load(ckpt)
        policy.load_state_dict(ckpt['policy_state_dict'])
        optimizer = optim.RAdam(policy.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        with open(os.path.join(log_path, 'train_log.json'), 'r') as f:
            log = json.load(f)
            min_loss = log['min_loss']
            best_ckpt = log['best_ckpt']
    else:
        print('Starting from scratch...')
        optimizer = optim.RAdam(policy.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
        min_loss = np.inf
        best_ckpt = None

    return policy, optimizer, min_loss, best_ckpt


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


def train(feature_dim: int = 512,
          seq_len: int = 8,
          batch_size: int = 64,
          epoch: int = 10,
          lr: float = 1e-3,
          weight_decay: float = 1e-5,
          loss_coeff: float = 0.9,
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
    train_set = ManiskillSeqDataset(env_id,
                                    obs_mode,
                                    control_mode,
                                    seq_len,
                                    train=True)
    val_set = ManiskillSeqDataset(env_id,
                                  obs_mode,
                                  control_mode,
                                  seq_len,
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

    policy, optimizer,  min_loss, best_ckpt = init_training(start_epoch,
                                                            state_dim,
                                                            act_dim,
                                                            feature_dim,
                                                            seq_len,
                                                            lr,
                                                            weight_decay)

    policy.train()

    criterion = nn.MSELoss(reduction='mean')

    print('Start training...')
    for epoch in tqdm(range(start_epoch+1, epoch+1)):
        total_train_loss = 0
        for state, action in train_loader:
            state = state.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            pred = policy(state)
            train_loss = criterion(pred, action)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()*len(state)

        mean_train_loss = total_train_loss / len(train_set)
        writer.add_scalar('Loss/Train', mean_train_loss, epoch)

        if epoch % ckpt_freq == 0:
            ckpt = save_ckpt(epoch, policy, optimizer)

            total_val_loss = validate(policy, val_loader, criterion)
            mean_val_loss = total_val_loss / len(val_set)
            writer.add_scalar('Loss/Validation', mean_val_loss, epoch)

            weighted_loss = mean_train_loss * loss_coeff + \
                mean_val_loss * (1 - loss_coeff)
            if weighted_loss < min_loss:
                min_loss = weighted_loss
                best_ckpt = ckpt

            log = dict(feature_dim=feature_dim,
                       seq_len=seq_len,
                       batch_size=batch_size,
                       lr=lr,
                       weight_decay=weight_decay,
                       loss_coeff=loss_coeff,
                       seed=seed,
                       min_loss=min_loss,
                       best_ckpt=best_ckpt)

            with open(os.path.join(log_path, 'train_log.json'), 'w') as f:
                json.dump(log, f, indent=4)

    writer.flush()
    writer.close()

    return best_ckpt


def test(ckpt: str,
         feature_dim: int = 256,
         seq_len: int = 8,
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

    feature_extractor = TransformerExtractor(state_dim=state_dim,
                                             feature_dim=feature_dim,
                                             seq_len=seq_len,
                                             layer_count=2)

    regressor = LinearRegressor(feature_dim=feature_dim,
                                act_dim=act_dim,
                                layer_count=3)

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
            state_deque = init_deque(obs, seq_len)
            state_sequence = np.array(state_deque, dtype=np.float32)
            while not terminated and not truncated:
                state_sequence = torch.from_numpy(state_sequence)
                state_sequence = state_sequence.unsqueeze(0).to(device)
                action = policy(state_sequence)
                action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action[0])
                state_sequence = update_deque(obs, state_deque)
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
                 seq_len: int,
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

    feature_extractor = TransformerExtractor(state_dim=state_dim,
                                             feature_dim=feature_dim,
                                             seq_len=seq_len,
                                             layer_count=2)

    regressor = LinearRegressor(feature_dim=feature_dim,
                                act_dim=act_dim,
                                layer_count=3)

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
            state_deque = init_deque(obs, seq_len)
            state_sequence = np.array(state_deque, dtype=np.float32)
            while not terminated and not truncated:
                state_sequence = torch.from_numpy(state_sequence)
                state_sequence = state_sequence.unsqueeze(0).to(device)
                action = policy(state_sequence)
                action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action[0])
                state_sequence = update_deque(obs, state_deque)
    env.close()


if __name__ == '__main__':
    seq_len = 8
    feature_dim = 256
    ckpt = train(lr=5e-4,
                 seq_len=seq_len,
                 feature_dim=feature_dim,
                 batch_size=128,
                 epoch=500,
                 ckpt_freq=5,
                 start_epoch=0)

    # ckpt = os.path.join(ckpt_path, 'ckpt_50.pt')

    success_seeds = test(ckpt,
                         feature_dim=feature_dim,
                         seq_len=seq_len,
                         num_episodes=50)
    # render_video(ckpt, success_seeds[:5])
