import os

import h5py
from torch.utils.data import Dataset

from utils.data import make_path


class GymDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        with h5py.File(dataset_path, 'r') as f:
            self.data = f['data']['demo_0']
            print(self.data)


if __name__ == '__main__':
    path = make_path(os.path.join('robomimic', 'datasets', 'can', 'ph', 'low_dim_v141.hdf5'))
    dataset = GymDataset(path)

