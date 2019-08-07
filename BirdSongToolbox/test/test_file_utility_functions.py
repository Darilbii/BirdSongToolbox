"""Test functions for the file_utility_functions """

import numpy as np
import pytest
from pathlib import Path
from shutil import rmtree
from BirdSongToolbox.file_utility_functions import _handle_data_path, _save_numpy_data, _load_numpy_data, \
    _save_pckl_data, _load_pckl_data, _save_json_data, _load_json_data


# TODO: Move Global Paths for Testing to a conftest.py file
@pytest.fixture(scope="module")
def file_utils_path(data_path):
    file_data_path = data_path / "test_file_utilities"

    yield file_data_path

    rmtree(file_data_path)


@pytest.fixture(scope="module")
def bird_id():
    return 'z020'


@pytest.fixture(scope="module")
def session():
    return 'day-2016-06-02'


@pytest.fixture(scope="module")
def goal_name():
    return 'test'


@pytest.fixture(scope="module")
def data_array():
    return np.arange(1, 100)


@pytest.mark.run(order=1)
def test_handle_data_path(file_utils_path, data_path, bird_id, session, goal_name):
    """ Test _handle_data_path """

    Goal = data_path / "test_file_utilities" / bird_id / session / goal_name
    path = _handle_data_path(data_name=goal_name, bird_id=bird_id, session=session, dir_path=file_utils_path,
                             make_parents=True)

    assert isinstance(path, Path)
    assert path == Goal


@pytest.mark.run(order=1)
def test_save_numpy_data(file_utils_path, data_path, bird_id, session, goal_name, data_array):
    """ Test _save_numpy_data """

    goal_name = goal_name + '.npy'
    Goal = data_path / "test_file_utilities" / bird_id / session / goal_name

    save_obj = _save_numpy_data(data=data_array, data_name=goal_name, bird_id=bird_id, session=session,
                                destination=file_utils_path, make_parents=False, verbose=True)

    assert Goal.exists()
    assert isinstance(save_obj, object)


@pytest.mark.run(order=1)
def test_load_numpy_data(file_utils_path,  bird_id, session, goal_name, data_array):
    """ Test _load_numpy_data """

    load_obj = _load_numpy_data(data_name=goal_name, bird_id=bird_id, session=session, source=file_utils_path,
                                verbose=True)

    assert isinstance(load_obj, (np.ndarray, tuple, dict, list))
    assert np.array_equal(load_obj, data_array)


@pytest.mark.run(order=1)
def test_save_pckl_data(file_utils_path, bird_id, session, goal_name, data_array):
    """ Test _save_pckl_data """

    save_obj = _save_pckl_data(data=data_array, data_name=goal_name, bird_id=bird_id, session=session,
                               destination=file_utils_path, make_parents=False, verbose=True)

    assert isinstance(save_obj, object)


@pytest.mark.run(order=1)
def test_load_pckl_data(file_utils_path, bird_id, session, goal_name, data_array):
    """ Test _load_pckl_data """

    load_obj = _load_pckl_data(data_name=goal_name, bird_id=bird_id, session=session, source=file_utils_path,
                               verbose=True)

    assert isinstance(load_obj, (np.ndarray, tuple, dict, list))
    assert np.array_equal(load_obj, data_array)


@pytest.mark.run(order=1)
@pytest.mark.parametrize("name,data", [("data_list", [1, 2, 3, 4, 5]),
                                       ("data_dict", {'1': [2, 6, 8], '3': 4, 'five': 6, '7': 'eight'}),
                                       ("data_dict_bad", {1: 2, 3: 4, 'five': 6, 7: 'eight'})])
def test_save_json_data(file_utils_path, bird_id, session, data, name):
    """ Test _save_json_data """

    _save_json_data(data=data, data_name=name, bird_id=bird_id, session=session,
                    destination=file_utils_path, make_parents=False, verbose=True)


@pytest.mark.run(order=1)
@pytest.mark.parametrize("name,data", [("data_list", [1, 2, 3, 4, 5]),
                                       ("data_dict", {'1': [2, 6, 8], '3': 4, 'five': 6, '7': 'eight'}),
                                       pytest.param("data_dict_bad", {1: 2, 3: 4, 'five': 6, 7: 'eight'},
                                                    marks=pytest.mark.xfail)])  # Json forces all keys to be str (Fails)
def test_load_json_data(file_utils_path, bird_id, session, data, name):
    """ Test _load_json_data """

    load_obj = _load_json_data(data_name=name, bird_id=bird_id, session=session, source=file_utils_path, verbose=True)

    assert isinstance(load_obj, (tuple, dict, list))
    assert load_obj == data
