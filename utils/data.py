import os


def make_path(*args):
    return os.path.join(os.getcwd(), *args)


if __name__ == '__main__':
    path = make_path(
        'RoboTurkPilot',
        'bins-Can',
        'low_dim.hdf5'
    )
    print(path)
