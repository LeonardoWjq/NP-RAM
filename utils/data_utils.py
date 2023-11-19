import os
import re

import clip
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_path(*args):
    return os.path.join(os.getcwd(), *args)


def read_xml(xml_path: str):
    with open(xml_path, 'r') as f:
        xml_string = f.read()
    return xml_string


def process_xml(xml_string: str):
    robosuite_path = make_path('robosuite')
    # return re.sub(r'/home/robot/installed_libraries/robosuite',
    #               robosuite_path,
    #               xml_string)
    return re.sub(r'/home/soroushn/code/robosuite-dev',
                  robosuite_path,
                  xml_string)


def swap_color(obs):
    group = obs['extra']
    group['cubeA_pose'][:], group['cubeB_pose'][:] = group['cubeB_pose'][:], group['cubeA_pose'][:]
    group['tcp_to_cubeA_pos'][:], group['tcp_to_cubeB_pos'][:
                                                            ] = group['tcp_to_cubeB_pos'][:], group['tcp_to_cubeA_pos'][:]
    group['cubeA_to_cubeB_pos'][:] = group['cubeA_to_cubeB_pos'][:] * -1


def encode_stack_cube_instruction(top_color='red', batch=500, seed=42):
    assert top_color in ['red', 'green']

    np.random.seed(seed)

    if top_color == 'red':
        file_path = make_path('instructions', 'red_on_green.txt')
    else:
        file_path = make_path('instructions', 'green_on_red.txt')

    with open(file_path, 'r') as f:
        instructions = f.readlines()

    tokens = clip.tokenize(instructions).to(device)
    model, _ = clip.load("RN50", device=device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features.cpu().numpy()

    sample_indices = np.random.choice(len(text_features), batch, replace=True)
    return text_features[sample_indices]


def flatten_obs(obs):
    '''
    Note: the order of the observations is important
    '''
    agent = obs['agent']
    extra = obs['extra']
    return np.concatenate([agent['qpos'],
                           agent['qvel'],
                           agent['base_pose'],
                           extra['tcp_pose'],
                           extra['cubeA_pose'],
                           extra['cubeB_pose'],
                           extra['tcp_to_cubeA_pos'],
                           extra['tcp_to_cubeB_pos'],
                           extra['cubeA_to_cubeB_pos']
                           ], axis=-1)


def obs_to_sequences(obs: np.array, sequence_len: int, mode: str = 'zero') -> np.array:
    assert mode in ['zero', 'repeat'], f'mode {mode} not supported'

    if mode == 'zero':
        head = np.zeros((sequence_len - 1, *obs[0].shape))
    else:
        head = np.repeat(obs[0][None], sequence_len - 1, axis=0)

    aug_obs = np.concatenate([head, obs], axis=0)
    sequences = []
    for i in range(len(obs)):
        sequences.append(aug_obs[i:i + sequence_len])
    return np.array(sequences)


if __name__ == '__main__':
    import h5py
    h5_path = make_path('datasets', 'train', 'trajectory_state.h5')
    f = h5py.File(h5_path, 'r')
    print(f['traj_0']['obs'])
    print(f['traj_0']['actions'])
    print(f['traj_0']['instruction'])
    f.close()
