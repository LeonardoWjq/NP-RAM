import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
from imitation.rewards.reward_nets import RewardNet
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from utils.wrappers import ContinuousTaskWrapper, RewardBonusWrapper


def get_space_len(space):
    if isinstance(space, Box):
        return space.shape[0]
    elif isinstance(space, Dict) or isinstance(space, dict):
        ls = []
        for key, value in space.items():
            ls.append(get_space_len(value))
        return np.sum(ls)
    else:
        raise ValueError(f'space is of type {type(space)}')


def make_env(env_id: str,
             obs_mode: str,
             reward_mode: str,
             control_mode: str,
             save_video: bool = False,
             max_episode_steps: int = None,
             record_dir: str = None,
             reward_net: RewardNet = None):

    def _init() -> gym.Env:
        env = gym.make(env_id,
                       obs_mode=obs_mode,
                       reward_mode=reward_mode,
                       control_mode=control_mode,
                       max_episode_steps=max_episode_steps,
                       render_mode="rgb_array")

        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env)

        if record_dir is not None:
            env = RecordEpisode(
                env=env,
                output_dir=record_dir,
                save_video=save_video,
                info_on_video=True,
                save_trajectory=True
            )

        if reward_net is not None:
            env = RewardBonusWrapper(env, reward_net)
        return env

    return _init


def make_vec_env(env_id,
                 num_envs,
                 obs_mode,
                 reward_mode,
                 control_mode,
                 save_video=False,
                 max_episode_steps=None,
                 record_dir=None,
                 reward_net=None):

    venv = DummyVecEnv([make_env(env_id,
                                 obs_mode,
                                 reward_mode,
                                 control_mode,
                                 save_video,
                                 max_episode_steps,
                                 record_dir,
                                 reward_net) for _ in range(num_envs)])
    venv = VecMonitor(venv)
    venv.reset()
    
    return venv