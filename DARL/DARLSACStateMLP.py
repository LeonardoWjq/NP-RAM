import json
import os

import mani_skill2.envs
import numpy as np
import torch
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from utils.data_utils import make_path
from utils.env_utils import make_vec_env

num_envs = 4
env_id = "LiftCube-v0"
obs_mode = "state"
control_mode = "pd_ee_delta_pose"
reward_mode = "normalized_dense"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def train(seed: int = 0,
          reward_net_state_dict: dict = None,
          reward_coeff: float = 0.01,
          total_steps: int = 640_000,
          train_episode_horizon: int = 50,
          policy_kwargs: dict = None,
          batch_size: int = 400,
          buffer_size: int = 40_000,
          gamma: float = 0.85,
          gradient_steps: int = 5,
          train_freq: int = 20,
          target_update_interval: int = 100,
          use_sde: bool = False,
          use_sde_at_warmup: bool = True,
          eval_frequency: int = 32_000,
          n_eval_episodes: int = 5,
          ckpt_frequency: int = 32_000,
          continue_training: bool = False):

    log_path = make_path('logs',
                         f'SAC-Bonus-{env_id}-{obs_mode}-{control_mode}',
                         f'seed-{seed}')
    ckpt_path = os.path.join(log_path, 'checkpoints')
    tb_path = os.path.join(log_path, 'tensorboard')
    eval_path = os.path.join(log_path, 'eval')
    best_model_path = os.path.join(log_path, 'best_model')

    eval_env = make_vec_env(env_id=env_id,
                            num_envs=1,
                            obs_mode=obs_mode,
                            reward_mode=reward_mode,
                            control_mode=control_mode,
                            save_video=True,
                            max_episode_steps=200,
                            record_dir=eval_path)

    reward_net = BasicShapedRewardNet(observation_space=eval_env.observation_space,
                                      action_space=eval_env.action_space,
                                      normalize_input_layer=RunningNorm,
                                      reward_hid_sizes=(256, 256),
                                      potential_hid_sizes=(256, 256))

    reward_net.load_state_dict(reward_net_state_dict)
    reward_net.to(device)

    train_env = make_vec_env(env_id=env_id,
                             num_envs=num_envs,
                             obs_mode=obs_mode,
                             reward_mode=reward_mode,
                             control_mode=control_mode,
                             save_video=False,
                             max_episode_steps=train_episode_horizon,
                             record_dir=None,
                             reward_net=reward_net,
                             reward_coeff=reward_coeff,
                             device=device)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=best_model_path,
                                 log_path=log_path,
                                 eval_freq=eval_frequency,
                                 deterministic=True,
                                 n_eval_episodes=n_eval_episodes)

    checkpoint_callback = CheckpointCallback(save_freq=ckpt_frequency,
                                             save_path=ckpt_path,
                                             name_prefix="sac_model",
                                             save_replay_buffer=True,
                                             save_vecnormalize=True,
                                             verbose=2)
    set_random_seed(seed)

    if continue_training:
        model = SAC.load(os.path.join(ckpt_path, 'latest_model'))
    else:
        model = SAC(policy='MlpPolicy',
                    env=train_env,
                    policy_kwargs=policy_kwargs,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    gamma=gamma,
                    gradient_steps=gradient_steps,
                    train_freq=train_freq,
                    target_update_interval=target_update_interval,
                    use_sde=use_sde,
                    use_sde_at_warmup=use_sde_at_warmup,
                    tensorboard_log=tb_path,
                    device=device,
                    verbose=1)
        
    model.learn(total_timesteps=total_steps,
                callback=[eval_callback, checkpoint_callback],
                reset_num_timesteps=False)

    model.save(os.path.join(ckpt_path, 'latest_model'))

    train_env.close()
    eval_env.close()


def test(seed: int = 1,
         model_path: str = None,
         evaluate_episodes: int = 50):

    log_path = make_path('logs',
                         f'SAC-Bonus-{env_id}-{obs_mode}-{control_mode}',
                         f'seed-{seed}')

    test_path = os.path.join(log_path, 'test')

    test_env = make_vec_env(env_id=env_id,
                            num_envs=1,
                            obs_mode=obs_mode,
                            reward_mode=reward_mode,
                            control_mode=control_mode,
                            save_video=True,
                            record_dir=test_path)

    test_env.seed(seed)
    test_env.reset()

    if model_path is None:
        best_model_path = os.path.join(log_path,
                                       'best_model',
                                       'best_model.zip')
        model = SAC.load(path=best_model_path,
                         env=test_env,
                         device=device)
    else:
        model = SAC.load(path=model_path,
                         env=test_env,
                         device=device)

    returns, ep_lens = evaluate_policy(model,
                                       test_env,
                                       deterministic=True,
                                       return_episode_rewards=True,
                                       n_eval_episodes=evaluate_episodes)

    # episode length < 200 means we solved the task before time ran out
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print(f"Success Rate: {success_rate}")
    print(f"Episode Lengths: {ep_lens}")
    print(f"Returns: {returns}")

    with open(os.path.join(test_path, "success.json"), "w") as f:
        log = dict(success_rate=f'success rate: {float(success_rate):.2%}')
        json.dump(log, f, indent=4)


if __name__ == '__main__':
    policy_kwargs = dict(net_arch=[256, 256])
    seeds = [3, 9, 7, 12, 4]
    for seed in seeds:
        reward_net_path = os.path.join('logs',
                                       f'GAIL-SAC-{env_id}-{obs_mode}-{control_mode}',
                                       f'seed-{seed}',
                                       'checkpoints',
                                       'reward_net_latest.pt')

        reward_net_state_dict = torch.load(reward_net_path)

        train(seed=seed,
              reward_net_state_dict=reward_net_state_dict,
              policy_kwargs=policy_kwargs)
        
        test(seed=seed)
