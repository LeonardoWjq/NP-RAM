import os
import re
from collections import OrderedDict

import clip
import h5py as h5
import numpy as np
import torch
from numpy.typing import NDArray
from typing import List
from imitation.data.types import Trajectory

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


def obs_to_sequences(obs: np.array, sequence_len: int, mode: str = 'zero') -> NDArray[np.float32]:
    assert mode in ['zero', 'repeat'], f'mode {mode} not supported'

    if mode == 'zero':
        head = np.zeros((sequence_len - 1, *obs[0].shape))
    else:
        head = np.repeat(obs[0][None], sequence_len - 1, axis=0)

    aug_obs = np.concatenate([head, obs], axis=0)
    sequences = []
    for i in range(len(obs)):
        sequences.append(aug_obs[i:i + sequence_len])
    return np.array(sequences, dtype=np.float32)


def segment_trajectory(traj: h5.Group, segment_size) -> List[Trajectory]:
    act = np.array(traj['actions'], dtype=np.float32)
    assert len(act) >= segment_size, f'segment size must be smaller than the length of the episode {len(act)}'

    obs = np.array(traj['obs'], dtype=np.float32)
    success = np.array(traj['success'])
    segments = []
    for i in range(len(obs)-segment_size):
        segments.append(Trajectory(obs=obs[i:i+segment_size+1],
                                   acts=act[i:i+segment_size],
                                   infos=None,
                                   terminal=success[i:i+segment_size].any()))
    
    return segments

def make_trajectory(data_path:str, segment_size: int = 64):
    trajs = []
    with h5.File(data_path, 'r') as f:
        for traj in f.values():
            trajs.extend(segment_trajectory(traj, segment_size))
    return trajs


def process_image(image: h5.Group):
    rgb_base = image["base_camera"]["rgb"]
    depth_base = image["base_camera"]["depth"]
    rgb_hand = image["hand_camera"]["rgb"]
    depth_hand = image["hand_camera"]["depth"]

    rgbd = np.concatenate([rgb_base, depth_base, rgb_hand, depth_hand],
                          axis=-1)
    return rgbd


def rescale_rgbd(rgbd: np.array, scale_rgb_only: bool = False):
    # rescales rgbd data
    rgb_base = rgbd[..., 0:3] / 255.0
    rgb_hand = rgbd[..., 4:7] / 255.0
    depth_base = rgbd[..., 3:4]
    depth_hand = rgbd[..., 7:8]
    if not scale_rgb_only:
        depth_base = rgbd[..., 3:4] / (2**10)
        depth_hand = rgbd[..., 7:8] / (2**10)
    return np.concatenate([rgb_base, depth_base, rgb_hand, depth_hand], axis=-1, dtype=np.float32)


def flatten_state(obs):
    if isinstance(obs, h5.Dataset):
        return np.array(obs, dtype=np.float32)
    elif isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    elif isinstance(obs, h5.Group) or isinstance(obs, OrderedDict) or isinstance(obs, dict):
        ls = []
        for key, value in obs.items():
            ls.append(flatten_state(value))
        return np.hstack(ls)
    else:
        raise ValueError(f'obs is of type {type(obs)}')


if __name__ == '__main__':
    h5_path = make_path('demonstrations',
                        'v0',
                        'rigid_body',
                        'LiftCube-v0',
                        'trajectory.state.pd_ee_delta_pose.h5')

    trajectories = make_trajectory(h5_path, 16)
    print(len(trajectories))
    # for traj in trajectories:
    #     print(len(traj))
