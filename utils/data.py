import os

import h5py

def make_path(path: str):
    return os.path.join(os.getcwd(), path)

if __name__ == '__main__':
    path = make_path(os.path.join('RoboTurkPilot', 'bins-Bread', 'demo.hdf5'))
    with h5py.File(path) as f:
        data = f['data']
        # print(list(data))
        group = data['demo_2']
        for key in group.keys():
            print(group[key])
        # print(state[:10])