from pathlib import Path
from source import database_handler as dh
from source.database_handler import DataManager
import pytest


def test_set_valid_data_path():
    new_path = str(Path.home())
    dh.set_data_path(str(Path.home()))
    assert dh.path_to_data == Path(new_path).absolute()


def test_set_invalid_data_path():
    new_path = 'this_is_not_a_path'
    with pytest.raises(Exception):
        assert dh.set_data_path(new_path)
