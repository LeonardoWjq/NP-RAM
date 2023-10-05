import os
import re
import numpy as np
import robosuite as suite

from utils.data import make_path

ROBOTURK_PATH = make_path('RoboTurkPilot')
# create environment instance
env = suite.make(
    env_name="PickPlace",  # try with other tasks like "Stack" and "Door"
    robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

xml_path = os.path.join(ROBOTURK_PATH, 'bins-Bread', 'models','model_1.xml')


with open(xml_path, 'r') as f:
    xml_string = f.read()

# print(xml_string)

# # reset the environment
env.reset_from_xml_string(xml_string)

# for i in range(1000):
#     action = np.random.randn(env.robots[0].dof) # sample random action
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     env.render()  # render on display
