
import os

import mani_skill2.envs
import numpy as np
import torch as torch
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet, RewardNet
from imitation.util.logger import configure
from imitation.util.networks import RunningNorm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO

from utils.data_utils import make_path, make_trajectory
from utils.env_utils import make_vec_env

num_envs = 4
env_id = "LiftCube-v0"
obs_mode = "state"
control_mode = "pd_ee_delta_pose"
reward_mode = "normalized_dense"
data_path = make_path('demonstrations',
                      'v0',
                      'rigid_body',
                      env_id,
                      f'trajectory.{obs_mode}.{control_mode}.h5')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def make_gail_callback(ckpt_path: str,
                       eval_path: str,
                       seed: int,
                       learner: PPO,
                       reward_net: RewardNet,
                       save_freq: int = 15,
                       eval_freq: int = 15,
                       eval_episodes: int = 10):

    def step_callback(step: int):
        if step % save_freq == 0:
            learner.save(os.path.join(ckpt_path, f"gail_ppo_{step}"))
            learner.save(os.path.join(ckpt_path, f"gail_ppo_latest"))
            torch.save(reward_net.state_dict(),
                       os.path.join(ckpt_path, f"reward_net_{step}.pt"))
            torch.save(reward_net.state_dict(),
                       os.path.join(ckpt_path, f"reward_net_latest.pt"))

        if step % eval_freq == 0:
            eval_env = make_vec_env(env_id=env_id,
                                    num_envs=1,
                                    obs_mode=obs_mode,
                                    reward_mode=reward_mode,
                                    control_mode=control_mode,
                                    record_dir=eval_path)
            eval_env.seed(seed)
            eval_env.reset()

            returns, ep_lens = evaluate_policy(learner,
                                               eval_env,
                                               deterministic=True,
                                               return_episode_rewards=True,
                                               n_eval_episodes=eval_episodes)
            success = np.array(ep_lens) < 200
            success_rate = success.mean()
            print(f"Success Rate: {success_rate}")
            print(f"Episode Lengths: {ep_lens}")
            print(f"Returns: {returns}")
            with open(os.path.join(eval_path, f"success.txt"), "a") as f:
                f.write(
                    f'step: {step} \t success rate: {float(success_rate):.2%}\n')

            eval_env.close()

    return step_callback


def train(seed: int = 0,
          total_time_steps: int = 32_000,
          train_episode_horizon: int = 50,
          segment_size: int = 32,
          demo_batch_size: int = 1024,
          gen_replay_buffer_capacity: int = 2048,
          disc_updates_per_round: int = 5,
          continue_training: bool = False):

    log_path = make_path('logs',
                         f'GAIL-PPO-{env_id}-{obs_mode}-{control_mode}',
                         f'seed-{seed}')
    ckpt_path = os.path.join(log_path, 'checkpoints')
    tb_path = os.path.join(log_path, 'tensorboard')
    eval_path = os.path.join(log_path, 'eval')

    set_random_seed(seed)

    # create num_envs training environments
    train_env = make_vec_env(env_id=env_id,
                             num_envs=num_envs,
                             obs_mode=obs_mode,
                             reward_mode=reward_mode,
                             control_mode=control_mode,
                             max_episode_steps=train_episode_horizon,
                             record_dir=None)

    # prepare trajectory segments
    demos = make_trajectory(data_path=data_path,
                            segment_size=segment_size)

    # create reward net
    reward_net = BasicShapedRewardNet(observation_space=train_env.observation_space,
                                      action_space=train_env.action_space,
                                      normalize_input_layer=RunningNorm,
                                      reward_hid_sizes=(256, 256),
                                      potential_hid_sizes=(256, 256))

    if continue_training:
        learner_ckpt = os.path.join(ckpt_path, "gail_ppo_latest")
        ppo_learner = PPO.load(learner_ckpt)
        reward_net_ckpt = os.path.join(ckpt_path, "reward_net_latest.pt")
        reward_net.load_state_dict(torch.load(reward_net_ckpt))
    else:
        # create our model from scratch
        ppo_learner = PPO("MlpPolicy",
                          env=train_env,
                          policy_kwargs=dict(net_arch=[256, 256]),
                          verbose=1,
                          n_steps=3200 // num_envs,
                          batch_size=400,
                          n_epochs=15,
                          tensorboard_log=tb_path,
                          gamma=0.85,
                          target_kl=0.05)

    # create GAIL trainer
    gail_trainer = GAIL(demonstrations=demos,
                        demo_batch_size=demo_batch_size,
                        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
                        n_disc_updates_per_round=disc_updates_per_round,
                        venv=train_env,
                        gen_algo=ppo_learner,
                        reward_net=reward_net,
                        allow_variable_horizon=False,
                        log_dir=log_path,
                        custom_logger=configure(log_path, ('tensorboard', 'stdout', 'csv')))

    # create callback
    callback = make_gail_callback(ckpt_path=ckpt_path,
                                  eval_path=eval_path,
                                  seed=seed,
                                  learner=ppo_learner,
                                  reward_net=reward_net)

    # train
    gail_trainer.train(total_timesteps=total_time_steps,
                       callback=callback)

    train_env.close()


def test(seed: int = 0,
         model_path: str = None,
         evaluate_episodes: int = 50):

    log_path = make_path('logs',
                         f'GAIL-PPO-{env_id}-{obs_mode}-{control_mode}',
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
        model_path = os.path.join(log_path,
                                  'checkpoints',
                                  'gail_ppo_latest')

    model = PPO.load(path=model_path,
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

    with open(os.path.join(test_path, "success.txt"), "a") as f:
        f.write(f'checkpoint: {model_path}\n')
        f.write(f'success rate: {float(success_rate):.2%}\n')


if __name__ == "__main__":
    seeds = [3, 9, 7, 12, 4]
    for seed in seeds:
        train(seed=seed, total_time_steps=400_000)
        test(seed=seed)
