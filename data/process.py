import json
import os

import h5py as h5
import numpy as np
from tqdm import tqdm

from utils.data_utils import (encode_stack_cube_instruction, flatten_obs,
                              make_path, swap_color)


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


def generate_train_val(split=0.9, seed=42):
    red_on_green_path = make_path('datasets', 'red_on_green')
    green_on_red_path = make_path('datasets', 'green_on_red')
    train_path = make_path('datasets', 'train')
    val_path = make_path('datasets', 'validation')

    red_on_green_fp = h5.File(os.path.join(red_on_green_path,
                                           'trajectory_state.h5'), 'r')
    green_on_red_fp = h5.File(os.path.join(green_on_red_path,
                                           'trajectory_state.h5'), 'r')
    train_fp = h5.File(os.path.join(train_path,
                                    'trajectory_state.h5'), 'w')
    val_fp = h5.File(os.path.join(val_path,
                                  'trajectory_state.h5'), 'w')

    rog_keys = list(red_on_green_fp.keys())
    rog_split = int(split * len(rog_keys))
    rog_train_keys = rog_keys[:rog_split]
    rog_val_keys = rog_keys[rog_split:]
    rog_encodings = encode_stack_cube_instruction(top_color='red',
                                                  batch=len(rog_keys),
                                                  seed=seed)
    rog_train_encodings = rog_encodings[:rog_split]
    rog_val_encodings = rog_encodings[rog_split:]

    print('Creating red on green datasets:')
    for key, encoding in tqdm(zip(rog_train_keys, rog_train_encodings)):
        observations = flatten_obs(red_on_green_fp[key]['obs'])
        actions = red_on_green_fp[key]['actions']
        traj = train_fp.create_group(key)
        traj.create_dataset('obs', data=observations[:-1])
        traj.create_dataset('actions', data=actions)
        traj.create_dataset('instruction', data=encoding)

    for key, encoding in tqdm(zip(rog_val_keys, rog_val_encodings)):
        observations = flatten_obs(red_on_green_fp[key]['obs'])
        actions = red_on_green_fp[key]['actions']
        traj = val_fp.create_group(key)
        traj.create_dataset('obs', data=observations[:-1])
        traj.create_dataset('actions', data=actions)
        traj.create_dataset('instruction', data=encoding)

    gor_keys = list(green_on_red_fp.keys())
    gor_split = int(split * len(gor_keys))
    gor_train_keys = gor_keys[:gor_split]
    gor_val_keys = gor_keys[gor_split:]
    gor_encodings = encode_stack_cube_instruction(top_color='green',
                                                  batch=len(gor_keys),
                                                  seed=seed)
    gor_train_encodings = gor_encodings[:gor_split]
    gor_val_encodings = gor_encodings[gor_split:]

    print('Creating green on red datasets:')
    for key, encoding in tqdm(zip(gor_train_keys, gor_train_encodings)):
        observations = flatten_obs(green_on_red_fp[key]['obs'])
        actions = green_on_red_fp[key]['actions']
        traj = train_fp.create_group(key)
        traj.create_dataset('obs', data=observations[:-1])
        traj.create_dataset('actions', data=actions)
        traj.create_dataset('instruction', data=encoding)

    for key, encoding in tqdm(zip(gor_val_keys, gor_val_encodings)):
        observations = flatten_obs(green_on_red_fp[key]['obs'])
        actions = green_on_red_fp[key]['actions']
        traj = val_fp.create_group(key)
        traj.create_dataset('obs', data=observations[:-1])
        traj.create_dataset('actions', data=actions)
        traj.create_dataset('instruction', data=encoding)

    red_on_green_fp.close()
    green_on_red_fp.close()
    train_fp.close()
    val_fp.close()

    train_trajs = rog_train_keys + gor_train_keys
    val_trajs = rog_val_keys + gor_val_keys

    with open(os.path.join(train_path, 'trajectory_state.json'), 'w') as f:
        json.dump(train_trajs, f, indent=4)
    with open(os.path.join(val_path, 'trajectory_state.json'), 'w') as f:
        json.dump(val_trajs, f, indent=4)


if __name__ == '__main__':
    create_and_split_color()
    generate_train_val()
