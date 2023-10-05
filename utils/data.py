import os

import h5py

def make_path(path: str):
    return os.path.join(os.getcwd(), path)

if __name__ == '__main__':
    path = make_path(os.path.join('RoboTurkPilot', 'bins-Bread', 'demo.hdf5'))
    with h5py.File(path) as f:
        print(f.values())