import json
import os
from typing import List

import gymnasium as gym
import mani_skill2.envs
import numpy as np
import torch
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.ppo import MlpPolicy
from tqdm import tqdm

from model.resnet import ResNetFeatureExtractor
from utils.data_utils import (flatten_state, make_path, process_image,
                              rescale_rgbd)
from utils.env_utils import get_state_len

ENV_ID = 'LiftCube-v0'
OBS_MODE = 'rgbd'
CONTROL_MODE = 'pd_ee_delta_pose'

gpu_id = 'cuda:1'
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')

log_path = make_path('logs', f'PPO-{ENV_ID}-{OBS_MODE}-{CONTROL_MODE}')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')
video_path = os.path.join(log_path, 'videos')
best_model_path = os.path.join(log_path, 'best_model')


class ObsWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        agnent_state_len = get_state_len(obs_space['agent'])
        extra_state_len = get_state_len(obs_space['extra'])
        state_dim = agnent_state_len + extra_state_len

        state_low = np.ones(state_dim) * -np.inf
        state_high = np.ones(state_dim) * np.inf

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
        self.state_dim = state_dim
        self.image_dim = (128, 128, 8)

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
                   reward_mode='normalized_dense',
                   max_episode_steps=250,
                   render_mode='rgb_array',
                   renderer_kwargs={'device': gpu_id})
    return ObsWrapper(env)


def make_venv(num_envs: int):
    venv = DummyVecEnv([make_env for _ in range(num_envs)])
    return venv


def train(num_envs: int = 4,
          gamma: float = 0.85,
          n_epochs: int = 15,
          rollout_steps: int = 3200,
          target_kl: float = 0.05,
          seed: int = 0,
          batch_size: int = 400,
          ckpt: str = None,
          steps_per_iter: int = 32000,
          total_iters: int = 10):

    eval_env = make_env()
    state_dim = eval_env.state_dim

    eval_env = RecordEpisode(eval_env,
                             video_path,
                             info_on_video=True,
                             save_trajectory=False)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecMonitor(eval_env)
    eval_env.seed(seed)
    eval_env.reset()

    venv = make_venv(num_envs)
    venv = VecMonitor(venv)
    venv.seed(seed)
    venv.reset()

    extractor = ResNetFeatureExtractor(state_dim=state_dim)
    policy_kwargs = dict(features_extractor_class=CustomExtractor,
                         features_extractor_kwargs=dict(resnet_extractor=extractor,
                                                        state_dim=state_dim),
                         net_arch=[256, 256])

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=best_model_path,
                                 log_path=log_path,
                                 eval_freq=steps_per_iter,
                                 deterministic=True,
                                 render=False)

    checkpoint_callback = CheckpointCallback(save_freq=steps_per_iter,
                                             save_path=ckpt_path,
                                             name_prefix="ppo",
                                             save_replay_buffer=True,
                                             save_vecnormalize=True,
                                             verbose=2
                                             )
    if ckpt is not None:
        extractor_ckpt = torch.load(ckpt)
        extractor.load_state_dict(
            extractor_ckpt['feature_extractor_state_dict'])
        policy_kwargs['features_extractor_kwargs']['training'] = False

    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=batch_size,
        gamma=gamma,
        n_epochs=n_epochs,
        n_steps=rollout_steps//num_envs,
        seed=seed,
        target_kl= target_kl,
        tensorboard_log=tb_path,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device
    )

    learner.learn(steps_per_iter*total_iters, callback=[checkpoint_callback, eval_callback])
    learner.save(os.path.join(ckpt_path, 'final_ckpt'))

    venv.close()

    return learner


def test(ckpt: str,
         num_episodes: int = 100) -> List[int]:

    env = make_env()

    learner = PPO.load(ckpt, device=device)

    success_seeds = []
    returns = {}

    with torch.no_grad():
        for seed in tqdm(range(num_episodes)):
            obs, _ = env.reset(seed=seed)
            G = 0
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action, _ = learner.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                G += reward

            if info['success']:
                success_seeds.append(seed)

            returns[seed] = G

    env.close()

    log = dict(returns=returns,
               num_episodes=num_episodes,
               success_rate=len(success_seeds) / num_episodes,
               success_seeds=success_seeds)

    with open(os.path.join(log_path, 'test_log.json'), 'w') as f:
        json.dump(log, f, indent=4)

    return success_seeds


def render_video(ckpt: str,
                 seed: int) -> None:

    env = make_env()

    env = RecordEpisode(
        env,
        video_path,
        info_on_video=True,
        save_trajectory=False
    )

    learner = PPO.load(ckpt, device=device)

    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False

    with torch.no_grad():
        while not terminated and not truncated:

            action, info = learner.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

    env.flush_video(suffix=f'{ENV_ID}_{OBS_MODE}_{CONTROL_MODE}_{seed}')
    env.close()


if __name__ == '__main__':
    train()
    # ckpt = os.path.join(ckpt_path, 'ckpt_100')
    # success_seeds = test(ckpt)
    # success_seeds = [65]
    # for seed in success_seeds:
    #     render_video(ckpt, seed=seed)
