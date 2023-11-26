import os

import gymnasium as gym
import gymnasium.spaces as spaces
import mani_skill2.envs
import numpy as np
import torch as torch
import torch.nn as nn
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from tqdm.notebook import tqdm

from utils.data_utils import make_path


# the following wrappers are importable from mani_skill2.utils.wrappers.sb3
# Defines a continuous, infinite horizon, task where done is always False
# unless a timelimit is reached.
class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, False, truncated, info

# A simple wrapper that adds a is_success key which SB3 tracks


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info


num_envs = 2
env_id = "LiftCube-v0"
obs_mode = "state"
control_mode = "pd_ee_delta_pose"
reward_mode = "normalized_dense"

log_path = make_path('logs', f'PPO-{env_id}-{obs_mode}-{control_mode}')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')
eval_video_path = os.path.join(log_path, 'eval/videos')
test_video_path = os.path.join(log_path, 'test/videos')
best_model_path = os.path.join(log_path, 'best_model')

# define an SB3 style make_env function for evaluation


def make_env(env_id: str,
             max_episode_steps: int = None,
             record_dir: str = None):
    def _init() -> gym.Env:
        env = gym.make(env_id,
                       obs_mode=obs_mode,
                       reward_mode=reward_mode,
                       control_mode=control_mode,
                       max_episode_steps=max_episode_steps,
                       render_mode="rgb_array")

        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env)
        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            env = RecordEpisode(
                env,
                record_dir,
                info_on_video=True
            )
        return env
    return _init


# create one eval environment
eval_env = DummyVecEnv([make_env(env_id, record_dir=eval_video_path) for i in range(1)])
eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
eval_env.seed(0)
eval_env.reset()

# create num_envs training environments
# we also specify max_episode_steps=50 to speed up training
train_env = DummyVecEnv([make_env(env_id, max_episode_steps=50) for i in range(num_envs)])
train_env = VecMonitor(train_env)
train_env.seed(0)
obs = train_env.reset()

eval_callback = EvalCallback(eval_env,
                             best_model_save_path=best_model_path,
                             log_path=log_path,
                             eval_freq=32000,
                             deterministic=True,
                             render=False)

checkpoint_callback = CheckpointCallback(save_freq=32000,
                                         save_path=ckpt_path,
                                         name_prefix="ppo_state",
                                         save_replay_buffer=True,
                                         save_vecnormalize=True,
                                         )

set_random_seed(0)  # set SB3's global seed to 0
rollout_steps = 3200

# create our model
policy_kwargs = dict(net_arch=[256, 256])

model = PPO("MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=rollout_steps // num_envs,
            batch_size=400,
            n_epochs=15,
            tensorboard_log=tb_path,
            gamma=0.85,
            target_kl=0.05
            )

model.learn(300_000, callback=[checkpoint_callback, eval_callback])
model.save(os.path.join(ckpt_path, "latest_model"))

eval_env.close()  # close the old eval env

# make a new one that saves to a different directory
test_env = DummyVecEnv([make_env(env_id, record_dir=test_video_path) for i in range(1)])
test_env = VecMonitor(test_env)  # attach this so SB3 can log reward metrics
test_env.seed(1)
test_env.reset()

returns, ep_lens = evaluate_policy(model,
                                   test_env,
                                   deterministic=True,
                                   render=False,
                                   return_episode_rewards=True,
                                   n_eval_episodes=10)
# episode length < 200 means we solved the task before time ran out
success = np.array(ep_lens) < 200
success_rate = success.mean()
print(f"Success Rate: {success_rate}")
print(f"Episode Lengths: {ep_lens}")
