import os

import h5py
import numpy as np

import robomimic
import robosuite as suite
from utils.data import make_path, read_xml, process_xml
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils


data_path = make_path('RoboTurkPilot', 'bins-full', 'low_dim.hdf5')
# data_path = make_path('robomimic','datasets','can','ph','low_dim_v141.hdf5')

env_meta = FileUtils.get_env_metadata_from_dataset(data_path)
env_meta['env_kwargs']['has_renderer'] = True
env = suite.make(env_name=env_meta['env_name'],
                 **env_meta['env_kwargs'])
# env = EnvUtils.create_env_from_metadata(
#     env_meta=env_meta,
#     render=False, 
#     render_offscreen=False,
#     use_image_obs=False
# )

with h5py.File(data_path, 'r') as f:
    data = f['data']
    demo = data['demo_2']
    states = demo['states']
    actions = demo['actions']
    xml_string = demo.attrs['model_file']
    env.reset_from_xml_string(process_xml(xml_string))
    env.sim.set_state_from_flattened(states[1])
    for action in actions:
        obs, reward, done, info = env.step(action)
        env.render()

env.close()



# model_path = make_path('RoboTurkPilot', 'bins-Can', 'models', 'model_1.xml')
# xml_string = read_xml(model_path)

# env.reset_from_xml_string(xml_string)

# with h5py.File(data_path, 'r') as f:
#     data = f['data']
#     demo = data['demo_1']

    # for action in actions:
    #     obs, reward, done, info = env.step(action)
    #     env.render()

# env.close()