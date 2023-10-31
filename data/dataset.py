import os

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import obs_to_sequences


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

        with h5py.File(dataset_path, 'r') as data:
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

        with h5py.File(dataset_path, 'r') as data:
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
    def __init__(self, train: bool = True, seq_len: int = 8) -> None:
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

        with h5py.File(dataset_path, 'r') as data:
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


if __name__ == '__main__':
    dataset = StackDatasetOriginalSequential(train=True)
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True, num_workers=0)
    for obs, actions in dataloader:
        print(obs.shape, actions.shape)
        break
