"""Test functions for the file_utility_functions """

from BirdSongToolbox.config.settings import DATA_DIR
from BirdSongToolbox.file_utility_functions import _handle_data_path, _save_numpy_data, _load_numpy_data, _save_pckl_data, _load_pckl_data, _save_json_data, _load_json_data
import numpy as np
import pytest
from pathlib import Path

#TODO: Move Global Paths for Testing to a conftest.py file
@pytest.fixture(scope="module")
def file_utils_path():
    data_path = DATA_DIR / "test_file_utilities"

    return data_path



@pytest.mark.run(order=1)
def test_handle_data_path(file_utils_path):
    """ Test handle_data_path """
    # data_name = 'data_name'
    goal_name = 'test'
    bird_id = 'z020'
    session = 'day-2016-06-02'
    # dir_path = None
    # make_parents = False

    something = _handle_data_path(data_name=goal_name, bird_id=bird_id, session=session, dir_path=file_utils_path, make_parents=True)

    Goal = DATA_DIR / "test_file_utilities" / bird_id / session / goal_name



@pytest.mark.run(order=1)
def test__save_numpy_data(file_utils_path):
    goal_name = 'test'
    bird_id = 'z020'
    session = 'day-2016-06-02'

    test_data = np.arange(1, 100)

    _save_numpy_data(data=test_data, data_name=goal_name, bird_id=bird_id, session=session, destination=file_utils_path,
                         make_parents=False)

    # Write test to Confirm that the file created exist and are in the expected location


