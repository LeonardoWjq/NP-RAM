import json
import os

import h5py as h5
import numpy as np
from tqdm import tqdm

from utils.data_utils import make_path, swap_color


def create_and_split_color(ratio=0.5, seed=42):
    '''
    generate two datasets from the original dataset, one with red on green and the other with green on red
    ratio: the ratio of the original dataset to be used for the red on green dataset
    '''
    assert 0 < ratio < 1
    assert isinstance(seed, int)
    np.random.seed(seed)

    source_path = make_path('datasets', 'trajectory_state_original.h5')
    red_on_green_path = make_path('datasets', 'red_on_green')
    green_on_red_path = make_path('datasets', 'green_on_red')

    with h5.File(source_path, 'r') as source:
        indices = np.arange(len(source.keys()))
        np.random.shuffle(indices)
        split = int(ratio * len(indices))
        red_on_green_indices = indices[:split]
        green_on_red_indices = indices[split:]

        with h5.File(os.path.join(red_on_green_path, 'trajectory_state.h5'), 'w') as dest:
            trajectories = []
            print('Creating red on green dataset:')
            for index in tqdm(red_on_green_indices):
                source.copy(f'traj_{index}', dest)
                trajectories.append(f'traj_{index}')

            with open(os.path.join(red_on_green_path, 'trajectory_state.json'), 'w') as f:
                json.dump(trajectories, f, indent=4)

        with h5.File(os.path.join(green_on_red_path, 'trajectory_state.h5'), 'w') as dest:
            trajectories = []
            print('Creating green on red dataset:')
            for index in tqdm(green_on_red_indices):
                source.copy(f'traj_{index}', dest)
                obs = dest[f'traj_{index}']['obs']
                swap_color(obs)
                trajectories.append(f'traj_{index}')

            with open(os.path.join(green_on_red_path, 'trajectory_state.json'), 'w') as f:
                json.dump(trajectories, f, indent=4)


if __name__ == '__main__':
    create_and_split_color()
