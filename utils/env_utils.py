from gymnasium.spaces import Box, Dict
import numpy as np


def get_state_len(obs_space):
    if isinstance(obs_space, Box):
        return obs_space.shape[0]
    elif isinstance(obs_space, Dict) or isinstance(obs_space, dict):
        ls = []
        for key, value in obs_space.items():
            ls.append(get_state_len(value))
        return np.sum(ls)
    else:
        raise ValueError(f'obs is of type {type(obs_space)}')
