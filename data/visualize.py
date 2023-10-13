import h5py

import robosuite as suite
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from utils.data_utils import make_path, process_xml


def visualize_traj(demo_num: int):
    assert isinstance(demo_num, int), 'demo_num must be an integer'
    assert demo_num >= 0 and demo_num < 200, 'demo_num must be between 0 and 199'

    data_path = make_path('robomimic', 'datasets', 'can',
                          'ph', 'low_dim_v141.hdf5')
    # data_path = make_path('RoboTurkPilot', 'bins-Can', 'low_dim.hdf5')
    env_meta = get_env_metadata_from_dataset(data_path)
    env_meta['env_kwargs']['has_renderer'] = True
    env = suite.make(env_name=env_meta['env_name'],
                     **env_meta['env_kwargs'])

    with h5py.File(data_path, 'r') as f:
        data = f['data']
        demo = data[f'demo_{demo_num}']
        states = demo['states']
        actions = demo['actions']
        xml_string = demo.attrs['model_file']
        env.reset_from_xml_string(process_xml(xml_string))
        env.sim.set_state_from_flattened(states[0])
        for action in actions:
            env.step(action)
            env.render()

    env.close()

if __name__ == '__main__':
    visualize_traj(199)