import os
import sys

path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
print(f'Adding {path_to_file} to sys env')
sys.path.insert(0, path_to_file)
