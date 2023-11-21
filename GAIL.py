import json
import math
import os
from pathlib import Path
from typing import List, Mapping, Optional, Union

import gymnasium as gym
import h5py
import mani_skill2.envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import (BasicRewardNet, NormalizedRewardNet,
                                           RewardNet)
from imitation.util.logger import configure
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.ppo import MlpPolicy
from torch.nn import (Flatten, Linear, TransformerEncoder,
                      TransformerEncoderLayer)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import StackDatasetOriginalSequential
from utils.data_utils import flatten_obs, make_path, obs_to_sequences
from utils.train_utils import init_deque, update_deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_path = make_path('logs', 'GAIL')
ckpt_path = os.path.join(log_path, 'checkpoints')
data_path = make_path('datasets', 'trajectory_state_original.h5')

Path(log_path).mkdir(exist_ok=True, parents=True)
Path(ckpt_path).mkdir(exist_ok=True, parents=True)


class SequenceWrapper(VecFrameStack):
    def __init__(self, venv: VecEnv, n_stack: int) -> None:
        super().__init__(venv, n_stack)
        low = np.reshape(self.observation_space.low,
                         (n_stack, *venv.observation_space.shape))
        high = np.reshape(self.observation_space.high,
                          (n_stack, *venv.observation_space.shape))
        self.observation_space = Box(low=low,
                                     high=high,
                                     dtype=self.observation_space.dtype)

    def step_wait(self):
        stackedobs, rewards, dones, infos = super().step_wait()
        reshaped_obs = np.reshape(stackedobs,
                                  (self.venv.num_envs,
                                   *self.observation_space.shape))
        return reshaped_obs, rewards, dones, infos

    def reset(self):
        stackedobs = super().reset()
        reshaped_obs = np.reshape(stackedobs,
                                  (self.venv.num_envs,
                                   *self.observation_space.shape))
        return reshaped_obs


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 100):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # [seq_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TransformerExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                 observation_space: Box,
                 seq_len: int,
                 obs_dim: int = 55,
                 features_dim: int = 256,
                 dropout: float = 0.1,
                 d_model: int = 128,
                 dim_ff: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4):
        super().__init__(observation_space, features_dim)

        self.d_model = d_model
        self.seq_len = seq_len
        self.obs_dim = obs_dim

        self.embedding = Linear(in_features=obs_dim,
                                out_features=d_model)  # project obs dimension to d_model dimension

        self.pos_encoder = PositionalEncoding(d_model=d_model,
                                              dropout=dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=num_heads,
                                                dim_feedforward=dim_ff,
                                                dropout=dropout,
                                                batch_first=True)  # define one layer of encoder multi-head attention

        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=num_layers)  # chain multiple layers of encoder multi-head attention

        self.flatten = Flatten(start_dim=1,
                               end_dim=-1)

        self.linear = Linear(in_features=d_model*seq_len,
                             out_features=features_dim)

        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = torch.reshape(observations,
                                     (-1, self.seq_len, self.obs_dim))
        embedding = self.embedding(observations)*math.sqrt(self.d_model)
        embedding = self.pos_encoder(embedding)
        feature = self.encoder(embedding)
        feature = self.flatten(feature)
        feature = self.linear(feature)
        return self.relu(feature)


class TransformerRewardNet(RewardNet):
    def __init__(self,
                 observation_space,
                 action_space,
                 seq_len: int,
                 obs_dim: int = 55,
                 features_dim: int = 256,
                 dropout: float = 0.1,
                 d_model: int = 128,
                 dim_ff: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4):

        super().__init__(observation_space,
                         action_space,
                         normalize_images=True)

        self.d_model = d_model
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.act_dim = action_space.shape[0]

        self.state_embedding = Linear(in_features=obs_dim,
                                      out_features=d_model)

        self.act_embedding = Linear(in_features=self.act_dim,
                                    out_features=d_model)

        self.done_embedding = Linear(in_features=1,
                                     out_features=d_model)

        self.pos_encoder = PositionalEncoding(d_model=d_model,
                                              dropout=dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=num_heads,
                                                dim_feedforward=dim_ff,
                                                dropout=dropout,
                                                batch_first=True)  # define one layer of encoder multi-head attention

        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=num_layers)  # chain multiple layers of encoder multi-head attention

        self.flatten = Flatten(start_dim=1,
                               end_dim=-1)

        self.linear = Linear(in_features=d_model*(seq_len+1),
                             out_features=features_dim)

        self.relu = nn.ReLU()

        self.output = Linear(in_features=features_dim,
                             out_features=1)

    def forward(self,
                state: torch.Tensor,  # (batch_size, *obs_shape)
                action: torch.Tensor,  # (batch_size, *action_shape)
                next_state: torch.Tensor,  # (batch_size, *obs_shape)
                done: torch.Tensor,  # (batch_size,)
                ) -> torch.Tensor:
        
        state = torch.reshape(state,
                              (-1, self.seq_len, self.obs_dim))

        state_embedding: torch.tensor = self.state_embedding(state)

        act_embedding: torch.tensor = self.act_embedding(action).unsqueeze(1)

        concat_embedding = torch.concat((state_embedding, act_embedding), dim=1)
        
        embedding = self.pos_encoder(concat_embedding)*math.sqrt(self.d_model)

        feature = self.encoder(embedding)
        feature = self.flatten(feature)
        feature = self.linear(feature)
        feature = self.relu(feature)

        return self.output(feature).squeeze(-1)


def prep_trajectory(data_path: str, seq_len: int, mode: str) -> List[Trajectory]:
    assert mode in ['zero', 'repeat'], f'mode {mode} not supported'

    traj_list = []
    with h5py.File(data_path, 'r') as f:
        for key in f.keys():
            traj = f[key]
            obs = flatten_obs(traj['obs'])
            sequences = obs_to_sequences(obs, seq_len)
            sequences = np.reshape(sequences, (sequences.shape[0], -1))
            acts = np.array(traj['actions'])
            traj = Trajectory(sequences, acts, infos=None, terminal=True)
            traj_list.append(traj)

    return traj_list


def record_video(agent: PPO,
                 seq_len: int,
                 seed: int = 42,
                 num_envs: int = 1,
                 max_length: int = 300,
                 suffix=''):
    
    agent.policy.eval()
    if suffix:
        prefix = f'PPO_StackCube_{suffix}'
    else:
        prefix = 'PPO_StackCube'

    venv = make_vec_env(
        "StackCube-v0",
        rng=np.random.default_rng(seed=seed),
        parallel=False,
        n_envs=num_envs,
        log_dir=log_path,
        max_episode_steps=max_length,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        env_make_kwargs=dict(obs_mode='state',
                             control_mode='pd_joint_delta_pos',
                             reward_mode='normalized_dense',
                             render_mode='rgb_array')
    )

    sequence_env: VecEnv = VecFrameStack(venv, n_stack=seq_len)
    rec_env = VecVideoRecorder(sequence_env,
                               os.path.join(log_path, 'video'),
                               record_video_trigger=lambda x: x == 0,
                               video_length=max_length,
                               name_prefix=prefix)

    obs = rec_env.reset()
    for step in range(max_length + 1):
        action, info = agent.predict(obs)
        obs, reward, done, info = rec_env.step(action)

    rec_env.close()
    agent.policy.train()


if __name__ == '__main__':
    SEED = 42
    SEQ_LEN = 8
    N_ENVS = 6
    trajectories = prep_trajectory(data_path, seq_len=SEQ_LEN, mode='zero')

    venv = make_vec_env(
        "StackCube-v0",
        rng=np.random.default_rng(seed=SEED),
        parallel=False,
        n_envs=N_ENVS,
        log_dir=log_path,
        max_episode_steps=300,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        env_make_kwargs=dict(obs_mode='state',
                             control_mode='pd_joint_delta_pos',
                             reward_mode='normalized_dense',
                             render_mode='rgb_array')
    )

    sequence_env: VecEnv = VecFrameStack(venv, n_stack=8)

    policy_kwargs = dict(features_extractor_class=TransformerExtractor,
                         features_extractor_kwargs=dict(seq_len=SEQ_LEN))

    learner = PPO(
        env=sequence_env,
        policy=MlpPolicy,
        batch_size=128,
        ent_coef=0.0,
        learning_rate=0.001,
        gamma=0.95,
        n_epochs=5,
        seed=SEED,
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    reward_net = TransformerRewardNet(observation_space=sequence_env.observation_space,
                                      action_space=sequence_env.action_space,
                                      seq_len=SEQ_LEN)

    reward_net = NormalizedRewardNet(reward_net,
                                     normalize_output_layer=RunningNorm)

    gail_trainer = GAIL(
        demonstrations=trajectories,
        demo_batch_size=512,
        gen_replay_buffer_capacity=1024,
        n_disc_updates_per_round=8,
        venv=sequence_env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        log_dir=log_path,
        init_tensorboard=True,
        init_tensorboard_graph=True,
        custom_logger=configure(log_path, ('tensorboard', 'stdout', 'csv'))
    )
    ckpts = 10
    step_per_ckpt = 300_000
    for ckpt in tqdm(range(1, ckpts+1)):
        gail_trainer.train(total_timesteps=step_per_ckpt)

        learner.save(os.path.join(ckpt_path,
                                  f'PPO_Agent_{ckpt*step_per_ckpt}'))

        torch.save(reward_net,
                   os.path.join(ckpt_path, f'Reward_Net_{ckpt*step_per_ckpt}'))

        record_video(agent=learner,
                     seq_len=SEQ_LEN,
                     seed=SEED,
                     num_envs=4,
                     max_length=400,
                     suffix=str(ckpt*step_per_ckpt)
                     )

    sequence_env.close()
