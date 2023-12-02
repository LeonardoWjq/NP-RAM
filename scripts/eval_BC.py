import os
from pathlib import Path
from typing import List

import gymnasium as gym
import mani_skill2.envs
import torch
from mani_skill2.utils.wrappers import RecordEpisode
from tqdm import tqdm

from model.base import BasePolicy
from utils.env_utils import get_space_len


def evaluate(env_id: str,
             obs_mode: str,
             control_mode: str,
             feature_extractor_class,
             feature_extractor_kwargs: dict,
             regressor_class,
             regressor_kwargs: dict,
             epoch: int,
             log_path: str,
             seed: int = 42,
             save_video: bool = False,
             num_episodes: int = 10,
             gpu_id: str = 'cuda:1') -> List[int]:

    eval_path = os.path.join(log_path, 'evaluation')
    Path(eval_path).mkdir(exist_ok=True, parents=True)
    ckpt = os.path.join(log_path,
                        'checkpoints',
                        f'ckpt_{epoch}.pt')
    device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')

    env = gym.make(id=env_id,
                   obs_mode=obs_mode,
                   control_mode=control_mode,
                   max_episode_steps=200,
                   render_mode="rgb_array",
                   enable_shadow=True,
                   renderer_kwargs={'device': gpu_id})

    env = RecordEpisode(
        env,
        eval_path,
        save_video=save_video,
        info_on_video=True,
        save_trajectory=True,
        save_on_reset=False
    )

    if obs_mode == 'state':
        state_dim = get_space_len(env.observation_space)
    elif obs_mode == 'rgbd':
        agent_dim = get_space_len(env.observation_space['agent'])
        extra_dim = get_space_len(env.observation_space['extra'])
        state_dim = agent_dim + extra_dim
    else:
        raise NotImplementedError

    act_dim = get_space_len(env.action_space)
    feature_extractor_kwargs['state_dim'] = state_dim
    regressor_kwargs['act_dim'] = act_dim

    env.reset(seed=seed)

    feature_extractor = feature_extractor_class(**feature_extractor_kwargs)
    regressor = regressor_class(**regressor_kwargs)
    policy = BasePolicy(feature_extractor, regressor)
    loaded_ckpt = torch.load(ckpt)
    policy.load_state_dict(loaded_ckpt['policy_state_dict'])
    policy.to(device)
    policy.eval()

    success_count = 0

    print('Start evaluating...')
    with torch.no_grad():
        for run in tqdm(range(1, num_episodes+1)):
            obs, _ = env.reset()
            terminated = False
            truncated = False

            while not terminated and not truncated:
                obs = torch.from_numpy(obs)
                obs = obs.unsqueeze(0).to(device)
                action = policy(obs)
                action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action[0])

            if info['success']:
                success_count += 1

            if save_video:
                if info['success']:
                    env.flush_video(
                        suffix=f'_seed{seed}_epoch{epoch}_run{run}_success')
                else:
                    env.flush_video(
                        suffix=f'_seed{seed}_epoch_{epoch}_run{run}_fail')
    env.close()

    success_rate = success_count / num_episodes
    print('Success Rate:', success_rate)

    with open(os.path.join(log_path, 'eval_log.text'), 'a') as f:
        f.write(f'Checkpoint: {ckpt}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Success Rate: {success_rate}\n')

    return success_rate
