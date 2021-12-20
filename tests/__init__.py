import os
import sys

path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
source_path = os.path.join(parent_path, 'source')
print(f'Adding {source_path} to sys env')


def import_source_modules():
    sys.path.insert(0, path_to_file)
