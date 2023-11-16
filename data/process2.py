import numpy as np
import h5py
from imitation.data.types import Trajectory
import os
from utils.data_utils import (encode_stack_cube_instruction, flatten_obs,
                              make_path, swap_color)


# Path to your .h5 file
dir_path = os.path.dirname(__file__)
data_path = os.path.join(dir_path, '..', 'datasets')
file_path = os.path.join(data_path, 'trajectory_state_original.h5') 

obs_list = []
acts_list = []
infos_list = []
traj_list = []

with h5py.File(file_path,'r') as file:
    for traj_key in file.keys():
        traj_data = file[traj_key]
        print(traj_data.keys())
        obs = flatten_obs(traj_data['obs'])
        acts = np.array(traj_data['actions'])
        traj = Trajectory(obs, acts, infos=None,terminal=True) 
        traj_list.append(traj)  
        # print(obs.shape) # (1c27, 55)
        # print(acts.shape) #(126,8)
        # print(traj_list)

print(len(traj_list))

# Function to extract and aggregate data from the .h5 file
# def aggregate_data(file_path):
#     obs_list = []
#     acts_list = []
#     infos_list = []

#     with h5py.File(file_path, 'r') as file:
#         for traj_key in file.keys():
#             traj_data = file[traj_key]
#             obs = np.array(traj_data['env_states'])
#             acts = np.array(traj_data['actions'])
#             infos = np.array([{} for _ in range(len(acts))])  # Dummy infos

#             obs_list.append(obs)
#             acts_list.append(acts)
#             infos_list.append(infos)

#     # Concatenate all trajectories
#     all_obs = np.concatenate(obs_list, axis=0)
#     all_acts = np.concatenate(acts_list, axis=0)
#     all_infos = np.concatenate(infos_list, axis=0)

#     return all_obs, all_acts, all_infos

# # Path to your .h5 file
# dir_path = os.path.dirname(__file__)
# data_path = os.path.join(dir_path, '..', 'datasets')
# file_path = os.path.join(data_path, 'trajectory_state_original.h5') 

# # Extract and aggregate data
# obs, acts, infos = aggregate_data(file_path)

# # Initialize TransitionsMinimal
# transitions = TransitionsMinimal(obs, acts, infos)

# # Now you can use 'transitions' in your machine learning model or analysis
