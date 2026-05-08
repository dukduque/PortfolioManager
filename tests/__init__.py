import os
import sys

path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
source_path = os.path.join(parent_path, 'source')

sys.path.insert(0, source_path)
sys.path.insert(0, path_to_file)


def import_source_modules():
    """No-op: source/ is now added to sys.path at import time via __init__."""
