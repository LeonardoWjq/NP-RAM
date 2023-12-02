from typing import Union

import gymnasium as gym
import torch
from imitation.rewards.reward_nets import RewardNet


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, False, truncated, info


class RewardBonusWrapper(gym.Wrapper):
    def __init__(self,
                 env: gym.Env,
                 reward_net: RewardNet,
                 device: Union[str, torch.device] = 'cuda',
                 bonus_coeff: float = 0.01) -> None:
        super().__init__(env)
        reward_net.eval()
        self.reward_net = reward_net
        self.device = device
        self.bonus_coeff = bonus_coeff

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = super().step(action)

        next_obs_tensor = torch.FloatTensor(
            next_obs).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        terminate_tensor = torch.FloatTensor([terminated]).to(self.device)

        with torch.no_grad():
            reward_bonus: torch.Tensor = self.reward_net(self.obs_tensor,
                                                         action_tensor,
                                                         next_obs_tensor,
                                                         terminate_tensor)

        reward += self.bonus_coeff * reward_bonus.cpu().numpy()
        self.obs_tensor = next_obs_tensor

        return next_obs, reward, terminated, truncated, info
