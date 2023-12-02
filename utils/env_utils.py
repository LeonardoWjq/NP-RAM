from gymnasium.spaces import Box, Dict
import numpy as np


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
