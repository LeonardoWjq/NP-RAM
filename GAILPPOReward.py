import json
import os
from typing import Any, List

import gymnasium as gym
import h5py as h5
import mani_skill2.envs
import numpy as np
import torch as torch
import torch.nn as nn
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet, RewardNet
from imitation.util.networks import RunningNorm
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from imitation.util.logger import configure
from utils.data_utils import make_path, make_trajectory

num_envs = 2
env_id = "LiftCube-v0"
obs_mode = "state"
control_mode = "pd_ee_delta_pose"
reward_mode = "normalized_dense"

log_path = make_path('logs', f'PPO-Shaping-{env_id}-{obs_mode}-{control_mode}')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')
eval_path = os.path.join(log_path, 'eval')
test_path = os.path.join(log_path, 'test')
best_model_path = os.path.join(log_path, 'best_model')

data_path = make_path('demonstrations',
                      'v0',
                      'rigid_body',
                      env_id,
                      f'trajectory.{obs_mode}.{control_mode}.h5'
                      )
gpu_id = 'cuda:0'
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, False, truncated, info


class RewardBonusWrapper(gym.Wrapper):
    def __init__(self,
                 env: gym.Env,
                 reward_net: RewardNet,
                 bonus_coeff: float = 0.01) -> None:
        super().__init__(env)
        reward_net.eval()
        self.reward_net = reward_net
        self.bonus_coeff = bonus_coeff

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = super().step(action)

        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
        terminate_tensor = torch.FloatTensor([terminated]).to(device)

        with torch.no_grad():
            reward_bonus: torch.Tensor = self.reward_net(self.obs_tensor, 
                                                         action_tensor, 
                                                         next_obs_tensor, 
                                                         terminate_tensor)

        reward += self.bonus_coeff * reward_bonus.cpu().numpy()
        self.obs_tensor = next_obs_tensor

        return next_obs, reward, terminated, truncated, info
    

def make_env(env_id: str,
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

        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
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


def make_gail_checkpoint_callback(learner: PPO,
                                  reward_net: RewardNet,
                                  save_freq: int = 5):
    def step_callback(step: int):
        if step % save_freq == 0:
            learner.save(os.path.join(ckpt_path, f"ppo_gail_{step}"))
            torch.save(reward_net.state_dict(),
                       os.path.join(ckpt_path, f"reward_net_{step}.pt"))

    return step_callback


def train_reward_net():
    # create one training environment
    train_env = DummyVecEnv([make_env(env_id,
                                      max_episode_steps=50) for _ in range(num_envs)])
    train_env = VecMonitor(train_env)
    train_env.reset()

    set_random_seed(0)

    demos = make_trajectory(data_path=data_path,
                            segment_size=32)

    rollout_steps = 3200

    # create our model
    policy_kwargs = dict(squash_output=True,
                         net_arch=[256, 256])

    ppo_learner = PPO("MlpPolicy",
                      train_env,
                      policy_kwargs=policy_kwargs,
                      verbose=1,
                      n_steps=rollout_steps // num_envs,
                      batch_size=400,
                      n_epochs=15,
                      tensorboard_log=tb_path,
                      gamma=0.85,
                      target_kl=0.05)

    # create reward net
    reward_net = BasicShapedRewardNet(observation_space=train_env.observation_space,
                                      action_space=train_env.action_space,
                                      normalize_input_layer=RunningNorm,
                                      reward_hid_sizes=(256, 256),
                                      potential_hid_sizes=(256, 256))

    gail_trainer = GAIL(demonstrations=demos,
                        demo_batch_size=1024,
                        gen_replay_buffer_capacity=512,
                        n_disc_updates_per_round=8,
                        venv=train_env,
                        gen_algo=ppo_learner,
                        reward_net=reward_net,
                        allow_variable_horizon=False,
                        log_dir=log_path,
                        custom_logger=configure(
                            log_path, ('tensorboard', 'stdout', 'csv'))
                        )

    gail_trainer.train(total_timesteps=320_000,
                       callback=make_gail_checkpoint_callback(ppo_learner, reward_net))

    train_env.close()
    return reward_net


def train_agent(reward_net: RewardNet):
    # create one eval environment
    eval_env = DummyVecEnv([make_env(env_id,
                                     record_dir=eval_path,
                                     save_video=True) for _ in range(1)])
    # attach this so SB3 can log reward metrics
    eval_env = VecMonitor(eval_env)
    eval_env.reset()

    reward_net.to(device)
    # create one training environment
    train_env = DummyVecEnv([make_env(env_id,
                                      max_episode_steps=50,
                                      reward_net=reward_net) for _ in range(num_envs)])
    train_env = VecMonitor(train_env)
    train_env.reset()

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=best_model_path,
                                 log_path=log_path,
                                 eval_freq=32_000,
                                 deterministic=True,
                                 n_eval_episodes=5)

    checkpoint_callback = CheckpointCallback(save_freq=32_000,
                                             save_path=ckpt_path,
                                             name_prefix="ppo_shaping",
                                             save_replay_buffer=True,
                                             save_vecnormalize=True,
                                             )

    set_random_seed(0)

    rollout_steps = 3200

    # create our model
    policy_kwargs = dict(squash_output=True,
                         net_arch=[256, 256])

    model = PPO("MlpPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                n_steps=rollout_steps // num_envs,
                batch_size=400,
                n_epochs=15,
                tensorboard_log=tb_path,
                gamma=0.85,
                target_kl=0.05,
                device=device)

    model.learn(total_timesteps=480_000,
                callback=[eval_callback, checkpoint_callback])
    model.save(os.path.join(ckpt_path, "ppo_shaping_latest_model"))

    train_env.close()
    eval_env.close()

    return model


def test(model: PPO):
    # make a new one that saves to a different directory
    test_env = DummyVecEnv([make_env(env_id,
                                     record_dir=test_path,
                                     save_video=True) for _ in range(1)])
    # attach this so SB3 can log reward metrics
    test_env = VecMonitor(test_env)
    test_env.seed(1)
    test_env.reset()

    returns, ep_lens = evaluate_policy(model,
                                       test_env,
                                       deterministic=True,
                                       render=False,
                                       return_episode_rewards=True,
                                       n_eval_episodes=50)

    # episode length < 200 means we solved the task before time ran out
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print(f"Success Rate: {success_rate}")
    print(f"Episode Lengths: {ep_lens}")

    with open(os.path.join(test_path, "success.json"), "w") as f:
        log = dict(success_rate=float(success_rate))
        json.dump(log, f, indent=4)


if __name__ == "__main__":
    reward_net = train_reward_net()
    ppo = train_agent(reward_net)
    test(ppo)
