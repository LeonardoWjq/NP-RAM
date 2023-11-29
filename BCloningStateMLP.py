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
                  weight_decay: float) -> Tuple[BasePolicy, optim.Optimizer, float, float, str]:

    feature_extractor = MLPExtractor(state_dim=state_dim,
                                     feature_dim=feature_dim,
                                     layer_count=3)
    regressor = LinearRegressor(feature_dim=feature_dim,
                                act_dim=act_dim)
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
            best_success_rate = log['best_success_rate']
            best_ckpt = log['best_ckpt']
    else:
        print('Starting from scratch...')
        optimizer = optim.RAdam(policy.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
        min_loss = np.inf
        best_success_rate = 0.0
        best_ckpt = None

    return policy, optimizer, min_loss, best_success_rate, best_ckpt


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
          loss_coeff: float = 0.9,
          seed: int = 42,
          ckpt_freq: int = 5,
          eval_freq: int = 50,
          eval_episodes: int = 20,
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

    policy, optimizer,  min_loss, best_success_rate, best_ckpt = init_training(start_epoch,
                                                                               state_dim,
                                                                               act_dim,
                                                                               feature_dim,
                                                                               lr,
                                                                               weight_decay)
    policy.train()
    dummy_state = torch.zeros((1, state_dim)).to(device)
    writer.add_graph(policy, dummy_state)

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

            writer.add_scalar('Loss/Weighted', weighted_loss, epoch)

            if weighted_loss < min_loss:
                min_loss = weighted_loss

            log = dict(feature_dim=feature_dim,
                       batch_size=batch_size,
                       lr=lr,
                       weight_decay=weight_decay,
                       loss_coeff=loss_coeff,
                       min_loss=min_loss,
                       best_success_rate=best_success_rate,
                       best_ckpt=best_ckpt)

            with open(os.path.join(log_path, 'train_log.json'), 'w') as f:
                json.dump(log, f, indent=4)

        if epoch % eval_freq == 0:
            ckpt = save_ckpt(epoch, policy, optimizer)
            success_rate = evalulate(ckpt,
                                     seed=seed,
                                     feature_dim=feature_dim,
                                     save_video=False,
                                     num_episodes=eval_episodes)
            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                best_ckpt = ckpt

    writer.flush()
    writer.close()

    return best_ckpt


def evalulate(ckpt: str,
              seed: int = 42,
              feature_dim: int = 256,
              save_video: bool = True,
              max_steps: int = 200,
              num_episodes: int = 10) -> List[int]:

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
        save_video=save_video,
        info_on_video=True,
        save_trajectory=True,
        save_on_reset=False
    )

    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    feature_extractor = MLPExtractor(state_dim=state_dim,
                                     feature_dim=feature_dim,
                                     layer_count=3)
    regressor = LinearRegressor(feature_dim=feature_dim,
                                act_dim=act_dim)
    policy = BasePolicy(feature_extractor=feature_extractor,
                        regressor=regressor,
                        squash_output=True)

    loaded_ckpt = torch.load(ckpt)
    policy.load_state_dict(loaded_ckpt['policy_state_dict'])
    policy.to(device)
    policy.eval()

    success_count = 0

    print('Start evaluating...')
    with torch.no_grad():
        for run in tqdm(range(1, num_episodes+1)):
            obs, _ = env.reset()
            terminated = False
            truncated = False

            while not terminated and not truncated:
                obs = torch.from_numpy(obs)
                obs = obs.unsqueeze(0).to(device)
                action = policy(obs)
                action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action[0])

            if info['success']:
                success_count += 1

            if save_video:
                if info['success']:
                    env.flush_video(
                        suffix=f'_{seed}_{run}_success')
                else:
                    env.flush_video(suffix=f'_{seed}_{run}_fail')
    env.close()

    success_rate = success_count / num_episodes
    print('Success Rate:', success_rate)

    with open(os.path.join(log_path, 'eval_log.text'), 'a') as f:
        f.write(f'Checkpoint: {ckpt}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Success Rate: {success_rate}\n')

    return success_rate


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
                                     layer_count=3)

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
    env = gym.make(id=env_id,
                   render_mode="rgb_array",
                   enable_shadow=True,
                   obs_mode=obs_mode,
                   control_mode=control_mode,
                   max_episode_steps=200,
                   renderer_kwargs={'device': gpu_id})
    env.reset()
    print(env._get_obs())
    # ckpt = train(lr=1e-3,
    #              batch_size=128,
    #              epoch=500,
    #              start_epoch=0)

    # ckpt = os.path.join(ckpt_path, 'ckpt_100.pt')

    # success_rate = evalulate(ckpt, num_episodes=50)
    # render_video(ckpt, success_seeds[:5])
