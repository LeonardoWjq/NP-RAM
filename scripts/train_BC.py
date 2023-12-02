import json
import os
from pathlib import Path
from typing import List, Tuple, Union

import mani_skill2.envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.utils import set_random_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import ManiskillSingleDataset
from model.base import BasePolicy, BaseRegressor
from model.mlp import BaseFeatureExtractor
from utils.data_utils import make_path
from scripts.eval_BC import evaluate


def validate(policy: BasePolicy,
             dataloader: DataLoader,
             criterion: nn.MSELoss,
             device: Union[str, torch.device]) -> float:

    total_loss = 0
    policy.eval()
    with torch.no_grad():
        for *obs, action in dataloader:
            for i in range(len(obs)):
                obs[i] = obs[i].to(device)

            action = action.to(device)
            pred = policy(*obs)
            loss = criterion(pred, action)
            total_loss += loss.item()*len(obs[0])
    policy.train()
    return total_loss


def init_training(start_epoch: int,
                  feature_extractor_class: BaseFeatureExtractor,
                  feature_extractor_kwargs: dict,
                  regressor_class: BaseRegressor,
                  regressor_kwargs: dict,
                  optimizer_class: optim.Optimizer,
                  optimizer_kwargs: dict,
                  device: Union[str, torch.device],
                  log_path: str = None) -> Tuple[BasePolicy, optim.Optimizer, float, float, str]:

    feature_extractor = feature_extractor_class(**feature_extractor_kwargs)
    regressor = regressor_class(**regressor_kwargs)
    policy = BasePolicy(feature_extractor=feature_extractor,
                        regressor=regressor,
                        squash_output=True)
    policy.to(device)

    if start_epoch > 0:
        assert log_path is not None, 'log_path must be provided if start_epoch > 0'
        ckpt_path = os.path.join(log_path, 'checkpoints')

        print('Loading checkpoint...')

        ckpt = os.path.join(ckpt_path, f'ckpt_{start_epoch}.pt')
        loaded_ckpt = torch.load(ckpt)

        policy.load_state_dict(loaded_ckpt['policy_state_dict'])

        optimizer = optimizer_class(
            params=policy.parameters(), **optimizer_kwargs)
        optimizer.load_state_dict(loaded_ckpt['optimizer_state_dict'])

        with open(os.path.join(log_path, 'train_log.json'), 'r') as f:
            log = json.load(f)
            min_loss = log['min_loss']
            best_success_rate = log['best_success_rate']
            best_ckpt = log['best_ckpt']
    else:
        print('Starting from scratch...')
        optimizer = optimizer_class(
            params=policy.parameters(), **optimizer_kwargs)
        min_loss = np.inf
        best_success_rate = 0.0
        best_ckpt = None

    return policy, optimizer, min_loss, best_success_rate, best_ckpt


def save_ckpt(log_path: str,
              epoch: int,
              policy: BasePolicy,
              optimizer: optim.Optimizer) -> str:
    '''
    save the policy and optimizer state dict
    returns the path to the checkpoint
    '''

    ckpt_path = os.path.join(log_path, 'checkpoints')
    Path(ckpt_path).mkdir(exist_ok=True, parents=True)

    ckpt = os.path.join(ckpt_path, f'ckpt_{epoch}.pt')
    torch.save(dict(policy_state_dict=policy.state_dict(),
                    optimizer_state_dict=optimizer.state_dict()),
               ckpt)
    return ckpt


def train(env_id: str,
          obs_mode: str,
          control_mode: str,
          feature_extractor_class: BaseFeatureExtractor,
          feature_extractor_kwargs: dict,
          regressor_class: BaseRegressor,
          regressor_kwargs: dict,
          optimizer_class: optim.Optimizer,
          optimizer_kwargs: dict,
          gpu_id: str = 'cuda:1',
          batch_size: int = 64,
          epoch: int = 100,
          loss_coeff: float = 0.9,
          seed: int = 42,
          ckpt_freq: int = 5,
          eval_freq: int = 10,
          eval_episodes: int = 20,
          start_epoch: int = 0) -> Tuple[int, str]:
    '''
    Train a policy using behavioral cloning.
    Use MLP feature extractor and a linear regressor.
    Output the best checkpoint.
    '''

    log_path = make_path('logs',
                         f'BC-{feature_extractor_class.get_name()}-{env_id}-{obs_mode}-{control_mode}',
                         f'seed-{seed}')
    tb_path = os.path.join(log_path, 'tensorboard')
    Path(tb_path).mkdir(exist_ok=True, parents=True)

    device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')

    print(f'Setting random seed to {seed}')
    set_random_seed(seed)

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
    feature_extractor_kwargs['state_dim'] = state_dim
    regressor_kwargs['act_dim'] = act_dim

    policy, optimizer,  min_loss, best_success_rate, best_ckpt = init_training(start_epoch=start_epoch,
                                                                               feature_extractor_class=feature_extractor_class,
                                                                               feature_extractor_kwargs=feature_extractor_kwargs,
                                                                               regressor_class=regressor_class,
                                                                               regressor_kwargs=regressor_kwargs,
                                                                               optimizer_class=optimizer_class,
                                                                               optimizer_kwargs=optimizer_kwargs,
                                                                               device=device,
                                                                               log_path=log_path)
    policy.train()

    try:
        dummy_state = torch.zeros((1, state_dim)).to(device)
        writer.add_graph(policy, dummy_state)
    except:
        print('Failed to add graph to tensorboard.')

    criterion = nn.MSELoss(reduction='mean')

    print('Start training...')
    for epoch in tqdm(range(start_epoch+1, epoch+1)):
        total_train_loss = 0
        for *obs, action in train_loader:
            for i in range(len(obs)):
                obs[i] = obs[i].to(device)
            action = action.to(device)

            optimizer.zero_grad()
            pred = policy(*obs)
            train_loss = criterion(pred, action)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()*len(obs[0])

        mean_train_loss = total_train_loss / len(train_set)
        writer.add_scalar('Loss/Train', mean_train_loss, epoch)

        if epoch % ckpt_freq == 0:
            save_ckpt(log_path, epoch, policy, optimizer)

            total_val_loss = validate(policy,
                                      val_loader,
                                      criterion,
                                      device)

            mean_val_loss = total_val_loss / len(val_set)
            writer.add_scalar('Loss/Validation', mean_val_loss, epoch)

            weighted_loss = mean_train_loss * loss_coeff + \
                mean_val_loss * (1 - loss_coeff)

            writer.add_scalar('Loss/Weighted', weighted_loss, epoch)

            if weighted_loss < min_loss:
                min_loss = weighted_loss

            log = dict(feature_extractor_kwargs=feature_extractor_kwargs,
                       regressor_kwargs=regressor_kwargs,
                       optimizer_kwargs=optimizer_kwargs,
                       batch_size=batch_size,
                       loss_coeff=loss_coeff,
                       min_loss=min_loss,
                       best_success_rate=best_success_rate,
                       best_ckpt=best_ckpt)

            with open(os.path.join(log_path, 'train_log.json'), 'w') as f:
                json.dump(log, f, indent=4)

        if epoch % eval_freq == 0:
            save_ckpt(log_path, epoch, policy, optimizer)

            success_rate = evaluate(env_id=env_id,
                                    obs_mode=obs_mode,
                                    control_mode=control_mode,
                                    feature_extractor_class=feature_extractor_class,
                                    feature_extractor_kwargs=feature_extractor_kwargs,
                                    regressor_class=regressor_class,
                                    regressor_kwargs=regressor_kwargs,
                                    epoch=epoch,
                                    log_path=log_path,
                                    seed=seed,
                                    num_episodes=eval_episodes,
                                    gpu_id=gpu_id)

            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                best_ckpt = epoch

    writer.flush()
    writer.close()

    return best_ckpt, log_path
