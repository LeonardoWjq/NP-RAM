import os
import re


def make_path(*args):
    return os.path.join(os.getcwd(), *args)


def read_xml(xml_path: str):
    with open(xml_path, 'r') as f:
        xml_string = f.read()
    return xml_string


def process_xml(xml_string: str):
    robosuite_path = make_path('robosuite')
    # return re.sub(r'/home/robot/installed_libraries/robosuite',
    #               robosuite_path,
    #               xml_string)
    return re.sub(r'/home/soroushn/code/robosuite-dev',
                  robosuite_path,
                  xml_string)


def swap_color(obs):
    group = obs['extra']
    group['cubeA_pose'][:], group['cubeB_pose'][:] = group['cubeB_pose'][:], group['cubeA_pose'][:]
    group['tcp_to_cubeA_pos'][:], group['tcp_to_cubeB_pos'][:] = group['tcp_to_cubeB_pos'][:], group['tcp_to_cubeA_pos'][:]
    group['cubeA_to_cubeB_pos'][:] = group['cubeA_to_cubeB_pos'][:] * -1
