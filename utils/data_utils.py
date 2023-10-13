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
