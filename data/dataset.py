import os

import h5py as h5
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.data_utils import (flatten_state, obs_to_sequences, process_image,
                              rescale_rgbd)

dir_path = os.path.dirname(__file__)
data_path = os.path.join(dir_path, '..', 'datasets')


class StackDataset(Dataset):
    def __init__(self, train: bool = True) -> None:
        super().__init__()
        if train:
            dataset_path = os.path.join(data_path,
                                        'train',
                                        'trajectory_state.h5')
        else:
            dataset_path = os.path.join(data_path,
                                        'validation',
                                        'trajectory_state.h5')

        self.instructions = []
        self.obs = []
        self.actions = []

        with h5.File(dataset_path, 'r') as data:
            for traj in data.values():
                obs = traj['obs'][:]
                actions = traj['actions'][:]
                ins = traj['instruction'][:]
                ins = np.repeat(ins[None], len(obs), axis=0)
                self.instructions.append(ins)
                self.obs.append(obs)
                self.actions.append(actions)

        self.instructions = np.concatenate(self.instructions, axis=0)
        self.obs = np.concatenate(self.obs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)

        assert len(self.instructions) == len(self.obs) == len(self.actions)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> tuple:
        return self.instructions[idx].astype('float32'), self.obs[idx], self.actions[idx]


class StackDatasetOriginal(Dataset):
    def __init__(self, train: bool = True) -> None:
        super().__init__()
        if train:
            dataset_path = os.path.join(data_path,
                                        'train',
                                        'trajectory_state_original.h5')
        else:
            dataset_path = os.path.join(data_path,
                                        'validation',
                                        'trajectory_state_original.h5')

        self.obs = []
        self.actions = []

        with h5.File(dataset_path, 'r') as data:
            for traj in data.values():
                obs = traj['obs'][:]
                actions = traj['actions'][:]
                self.obs.append(obs)
                self.actions.append(actions)

        self.obs = np.concatenate(self.obs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)

        assert len(self.obs) == len(self.actions)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> tuple:
        return self.obs[idx], self.actions[idx]


class StackDatasetOriginalSequential(Dataset):
    def __init__(self, seq_len: int, train: bool = True) -> None:
        super().__init__()
        if train:
            dataset_path = os.path.join(data_path,
                                        'train',
                                        'trajectory_state_original.h5')
        else:
            dataset_path = os.path.join(data_path,
                                        'validation',
                                        'trajectory_state_original.h5')

        self.obs = []
        self.actions = []

        with h5.File(dataset_path, 'r') as data:
            for traj in data.values():
                obs = np.array(traj['obs'][:])
                sequences = obs_to_sequences(obs, seq_len)
                actions = traj['actions'][:]
                self.obs.append(sequences)
                self.actions.append(actions)

        self.obs = np.concatenate(self.obs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)

        assert len(self.obs) == len(self.actions)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> tuple:
        return self.obs[idx], self.actions[idx]


class ManiskillDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 load_count: int = None) -> None:
        super().__init__()

        self.states = []
        self.rgbds = []
        self.actions = []

        with h5.File(data_path, 'r') as data:
            self.keys = list(data.keys())

        if load_count is not None:
            load_keys = self.keys[:load_count]
        else:
            load_keys = self.keys

        with h5.File(data_path, 'r') as data:
            for key in tqdm(load_keys):
                traj = data[key]
                obs = traj['obs']

                state = np.hstack([flatten_state(obs['agent']), flatten_state(obs['extra'])])
                self.states.append(state[:-1])

                act = traj['actions']
                self.actions.append(np.array(act, dtype=np.float32))

                image = obs['image']
                rgbd = process_image(image)
                rescaled_rgbd = rescale_rgbd(rgbd)
                self.rgbds.append(rescaled_rgbd[:-1])

        self.states = np.vstack(self.states)
        self.rgbds = np.vstack(self.rgbds)
        self.actions = np.vstack(self.actions)

        assert len(self.states) == len(self.rgbds) == len(self.actions)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple:
        state = torch.from_numpy(self.states[idx])
        rgbd = torch.from_numpy(self.rgbds[idx])
        action = torch.from_numpy(self.actions[idx])
        return state, rgbd, action


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_path = os.path.join(dir_path,
                             '..',
                             'demonstrations',
                             'v0',
                             'rigid_body',
                             'LiftCube-v0',
                             'trajectory.rgbd.pd_ee_delta_pose.h5'
                             )
    dataset = ManiskillDataset(data_path, load_count=1)
    print(len(dataset.keys))
    dataloader = DataLoader(dataset, batch_size=32)
    for state, rgbd, action in dataloader:
        print(state.shape)
        print(rgbd.shape)
        print(action.shape)
        break
