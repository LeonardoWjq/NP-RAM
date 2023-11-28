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
from model.resnet import ResNetExtractor
from model.linear import LinearRegressor
from model.base import BasePolicy
from utils.data_utils import make_path, rescale_rgbd, process_image, flatten_state

env_id = 'LiftCube-v0'
obs_mode = 'rgbd'
control_mode = 'pd_ee_delta_pose'


log_path = make_path('logs', f'BC-ResNet-{env_id}-{obs_mode}-{control_mode}')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')
video_path = os.path.join(log_path, 'videos')

Path(log_path).mkdir(exist_ok=True, parents=True)
Path(ckpt_path).mkdir(exist_ok=True, parents=True)
Path(tb_path).mkdir(exist_ok=True, parents=True)
Path(video_path).mkdir(exist_ok=True, parents=True)

gpu_id = 'cuda:2'
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')


def validate(policy: BasePolicy,
             dataloader: DataLoader,
             criterion: nn.MSELoss) -> float:
    total_loss = 0
    policy.eval()
    with torch.no_grad():
        for state, rgbd, action in dataloader:
            state = state.to(device)
            rgbd = rgbd.to(device)
            action = action.to(device)

            pred = policy(state, rgbd)
            loss = criterion(pred, action)
            total_loss += loss.item()*len(state)
    policy.train()
    return total_loss


def init_training(start_epoch: int,
                  state_dim: int,
                  act_dim: int,
                  lr: float,
                  weight_decay: float) -> Tuple[BasePolicy, optim.Optimizer, float, str]:

    feature_extractor = ResNetExtractor(state_dim=state_dim)
    feature_dim = feature_extractor.get_feature_dim()
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


def train(batch_size: int = 64,
          epoch: int = 10,
          lr: float = 1e-3,
          weight_decay: float = 1e-5,
          loss_coeff: float = 0.7,
          seed: int = 42,
          ckpt_freq: int = 5,
          start_epoch: int = 0) -> str:
    '''
    Train a policy using behavioral cloning.
    Use ResNet feature extractor and a linear regressor.
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

    policy, optimizer, min_loss, best_ckpt = init_training(start_epoch,
                                                           state_dim,
                                                           act_dim,
                                                           lr,
                                                           weight_decay)
    policy.train()
    dummy_state = torch.zeros((1, state_dim)).to(device)
    dummy_rgbd = torch.zeros((1, 128, 128, 8)).to(device)
    writer.add_graph(policy, (dummy_state, dummy_rgbd))

    criterion = nn.MSELoss(reduction='mean')

    print('Start training...')
    for epoch in tqdm(range(start_epoch+1, epoch+1)):
        total_train_loss = 0
        for state, rgbd, action in train_loader:
            state = state.to(device)
            rgbd = rgbd.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            pred = policy(state, rgbd)
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

            weighted_loss = loss_coeff*mean_train_loss + \
                (1-loss_coeff)*mean_val_loss
            if weighted_loss < min_loss:
                min_loss = weighted_loss
                best_ckpt = ckpt

            log = dict(batch_size=batch_size,
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

    obs, _ = env.reset()
    state = np.hstack([flatten_state(obs['agent']),
                       flatten_state(obs['extra'])])
    state_dim = state.shape[0]
    act_dim = env.action_space.shape[0]

    feature_extractor = ResNetExtractor(state_dim=state_dim)
    feature_dim = feature_extractor.get_feature_dim()
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

            while not terminated and not truncated:
                state = np.hstack([flatten_state(obs['agent']),
                                   flatten_state(obs['extra'])])

                rgbd = process_image(obs['image'])
                # the environment already rescales the depth
                rgbd = rescale_rgbd(rgbd, scale_rgb_only=True)

                state = torch.from_numpy(state)
                rgbd = torch.from_numpy(rgbd)
                state = state.unsqueeze(0).to(device)
                rgbd = rgbd.unsqueeze(0).to(device)

                action = policy(state, rgbd)
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
    obs, _ = env.reset()
    state = np.hstack([flatten_state(obs['agent']),
                       flatten_state(obs['extra'])])
    state_dim = state.shape[0]
    act_dim = env.action_space.shape[0]

    feature_extractor = ResNetExtractor(state_dim=state_dim)
    feature_dim = feature_extractor.get_feature_dim()
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

            while not terminated and not truncated:
                state = np.hstack([flatten_state(obs['agent']),
                                   flatten_state(obs['extra'])])
                rgbd = process_image(obs['image'])
                rgbd = rescale_rgbd(rgbd, scale_rgb_only=True)
                state = torch.from_numpy(state)
                rgbd = torch.from_numpy(rgbd)
                state = state.unsqueeze(0).to(device)
                rgbd = rgbd.unsqueeze(0).to(device)

                action = policy(state, rgbd)
                action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action[0])
    env.close()


if __name__ == '__main__':

    ckpt = train(lr=1e-3,
                 batch_size=128,
                 epoch=400,
                 ckpt_freq=4,
                 start_epoch=0)

    # ckpt = os.path.join(ckpt_path, 'ckpt_32.pt')

    success_seeds = test(ckpt=ckpt,
                         num_episodes=50)
    # render_video(ckpt=ckpt,
    #              seeds=success_seeds[:5])
