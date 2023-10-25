import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utils.data_utils import make_path


class StackDataset(Dataset):
    def __init__(self, train: bool = True) -> None:
        super().__init__()
        if train:
            dataset_path = make_path(
                'datasets', 'train', 'trajectory_state.h5')
        else:
            dataset_path = make_path(
                'datasets', 'validation', 'trajectory_state.h5')

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


if __name__ == '__main__':
    dataset = StackDataset()
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True, num_workers=4)
    for ins, obs, actions in dataloader:
        print(ins.shape, obs.shape, actions.shape)
        break
