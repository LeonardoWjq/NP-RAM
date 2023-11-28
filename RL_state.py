import json
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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from model.mlp import MLPExtractor
from utils.data_utils import make_path


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, False, truncated, info


num_envs = 2
env_id = "LiftCube-v0"
obs_mode = "state"
control_mode = "pd_ee_delta_pose"
reward_mode = "normalized_dense"

log_path = make_path('logs', f'PPO-BC-FREEZE-{env_id}-{obs_mode}-{control_mode}')
ckpt_path = os.path.join(log_path, 'checkpoints')
tb_path = os.path.join(log_path, 'tensorboard')
eval_path = os.path.join(log_path, 'eval')
test_path = os.path.join(log_path, 'test')
best_model_path = os.path.join(log_path, 'best_model')

gpu_id = 'cuda:1'
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')


def make_env(env_id: str,
             save_video: bool = False,
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
            env = RecordEpisode(
                env=env,
                output_dir=record_dir,
                save_video=save_video,
                info_on_video=True,
                save_trajectory=True
            )
        return env
    return _init


class LiftCubeMLP(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 features_dim: int = 256,
                 freeze: bool = False
                 ) -> None:
        super().__init__(observation_space, features_dim)
        state_dim = observation_space.shape[0]
        self.mlp = MLPExtractor(state_dim, features_dim)
        self.freeze = freeze

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feature = self.mlp(obs)
        if self.freeze:
            feature = feature.detach()
        return feature


def train(extractor_state_dict: dict = None,
          freeze_extractor: bool = False):
    # create one eval environment
    eval_env = DummyVecEnv([make_env(env_id,
                                     record_dir=eval_path,
                                     save_video=True) for _ in range(1)])
    # attach this so SB3 can log reward metrics
    eval_env = VecMonitor(eval_env)
    eval_env.reset()
    # create one training environment
    train_env = DummyVecEnv([make_env(env_id,
                                      max_episode_steps=50) for _ in range(num_envs)])
    train_env = VecMonitor(train_env)
    train_env.reset()

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=best_model_path,
                                 log_path=log_path,
                                 eval_freq=32000,
                                 deterministic=True,
                                 n_eval_episodes=5)

    checkpoint_callback = CheckpointCallback(save_freq=32000,
                                             save_path=ckpt_path,
                                             name_prefix="ppo_state",
                                             save_replay_buffer=True,
                                             save_vecnormalize=True,
                                             )

    set_random_seed(0)
    rollout_steps = 3200

    # create our model
    policy_kwargs = dict(squash_output=True,
                         activation_fn=nn.Mish,
                         net_arch=[],
                         features_extractor_class=LiftCubeMLP,
                         features_extractor_kwargs=dict(freeze=freeze_extractor))

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
                device=device
                )
    
    if extractor_state_dict is not None:
        model.policy.features_extractor.mlp.load_state_dict(extractor_state_dict)

    model.learn(320_000, callback=[checkpoint_callback, eval_callback])
    model.save(os.path.join(ckpt_path, "latest_model"))

    train_env.close()  # close the training en
    eval_env.close()  # close the old eval env
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
                                       n_eval_episodes=30)

    # episode length < 200 means we solved the task before time ran out
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print(f"Success Rate: {success_rate}")
    print(f"Episode Lengths: {ep_lens}")

    with open(os.path.join(test_path, "success.json"), "w") as f:
        log = dict(success_rate=float(success_rate))
        json.dump(log, f, indent=4)


if __name__ == "__main__":
    extractor_ckpt_path = make_path('logs',
                                    'BC-LiftCube-v0-state-pd_ee_delta_pose',
                                    'checkpoints',
                                    'ckpt_272.pt')
    extractor_ckpt_path = torch.load(extractor_ckpt_path)
    extractor_state_dict = extractor_ckpt_path['feature_extractor_state_dict']
    model = train(extractor_state_dict=extractor_state_dict,
                  freeze_extractor=True)
    test(model)
