import os

import gymnasium as gym
import mani_skill2.envs
import numpy as np
import torch
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from model.resnet import ResNetFeatureExtractor
from utils.data_utils import (flatten_state, make_path, process_image,
                              rescale_rgbd)
from utils.env_utils import get_state_len
from tqdm import tqdm

ENV_ID = 'LiftCube-v0'
OBS_MODE = 'rgbd'
CONTROL_MODE = 'pd_ee_delta_pose'

gpu_id = 'cuda:1'
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')

log_path = make_path('logs', f'PPO-{ENV_ID}-{OBS_MODE}-{CONTROL_MODE}')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')
video_path = os.path.join(log_path, 'videos')


class ObsWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        agnent_state_len = get_state_len(obs_space['agent'])
        extra_state_len = get_state_len(obs_space['extra'])
        state_len = agnent_state_len + extra_state_len

        state_low = np.ones(state_len) * -np.inf
        state_high = np.ones(state_len) * np.inf

        rgbd_low = np.zeros(128*128*8)

        base_rgbd_high = np.ones(128*128*3)
        base_depth_high = np.ones(128*128)*np.inf
        hand_rgbd_high = np.ones(128*128*3)
        hand_depth_high = np.ones(128*128)*np.inf
        rgbd_high = np.hstack([base_rgbd_high,
                               base_depth_high,
                               hand_rgbd_high,
                               hand_depth_high]
                              )

        low = np.hstack([state_low, rgbd_low])
        high = np.hstack([state_high, rgbd_high])

        obs_space = gym.spaces.Box(low=low,
                                   high=high,
                                   shape=low.shape,
                                   dtype=np.float32)

        self.observation_space = obs_space

    def observation(self, obs: dict):
        state = np.hstack([flatten_state(obs['agent']),
                           flatten_state(obs['extra'])])
        image = obs['image']
        rgbd = process_image(image)
        # the environment already scales depth
        rgbd: np.ndarray = rescale_rgbd(rgbd, scale_rgb_only=True)
        return np.hstack([state, rgbd.flatten()])


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: Box,
                 resnet_extractor: ResNetFeatureExtractor,
                 state_dim: int,
                 training: bool = True
                 ):
        features_dim = resnet_extractor.get_out_dim()
        super().__init__(observation_space, features_dim)
        self.extractor = resnet_extractor
        self.state_dim = state_dim
        self.training = training
        if training:
            self.extractor.train()
        else:
            self.extractor.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state = obs[:, :self.state_dim]
        image = obs[:, self.state_dim:]
        image = image.reshape(-1, 128, 128, 8)

        feature = self.extractor(state, image)

        if not self.training:
            feature = feature.detach()

        return feature


def make_env():
    env = gym.make(id=ENV_ID,
                   obs_mode=OBS_MODE,
                   control_mode=CONTROL_MODE,
                   max_episode_steps=250,
                   renderer_kwargs={'device': gpu_id})
    return ObsWrapper(env)


def make_venv(num_envs: int):
    venv = DummyVecEnv([make_env for _ in range(num_envs)])
    return venv


def train(num_envs: int = 4,
          lr: float = 0.001,
          gamma: float = 0.95,
          n_epochs: int = 5,
          seed: int = 42,
          batch_size: int = 128,
          ckpt: str = None,
          steps_per_iter: int = 10000,
          total_iters: int = 100):
    
    venv = make_venv(num_envs)
    state_dim = venv.observation_space.shape[0] - 128*128*8
    extractor = ResNetFeatureExtractor(state_dim=state_dim)
    policy_kwargs = dict(features_extractor_class=CustomExtractor,
                         features_extractor_kwargs=dict(resnet_extractor=extractor,
                                                        state_dim=state_dim))
    if ckpt is not None:
        extractor_ckpt = torch.load(ckpt)
        extractor.load_state_dict(extractor_ckpt['feature_extractor_state_dict'])
        policy_kwargs['features_extractor_kwargs']['training'] = False


    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=batch_size,
        ent_coef=0.0,
        learning_rate=lr,
        gamma=gamma,
        n_epochs=n_epochs,
        seed=seed,
        tensorboard_log=tb_path,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device
    )

    for iter in tqdm(range(1, total_iters+1)):
        learner.learn(total_timesteps=steps_per_iter,
                      reset_num_timesteps=False,
                      progress_bar=False)
        learner.save(os.path.join(ckpt_path, f'ckpt_{iter}'))

    venv.close()

    return learner


if __name__ == '__main__':
    train()
