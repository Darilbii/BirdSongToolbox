from BirdSongToolbox.config.settings import INTERMEDIATE_DATA_PATH

import numpy as np
import json
import pickle
from pathlib import Path


def _handle_data_path(data_name: str, bird_id: str, session: str, dir_path=None, make_parents=False):
    """

    :param data_name:
    :param bird_id:
    :param session:
    :param dir_path:  Path(), optional
        Destination of data to be save other than the default intermediate location
    :param make_parents: bool, optional
        if True, it will create all of the parent folders for the Data File
    :return:
    """

    # Handle Use Cases of the dir_path variable
    if dir_path is not None:
        if isinstance(dir_path, str):
            data_path = Path(dir_path)  # Make a Path instance
        elif isinstance(dir_path, Path()):
            data_path = dir_path  # already a Path instance
        else:
            raise TypeError  # Currently only suppor str and Path()
    else:
        if not INTERMEDIATE_DATA_PATH.exists():
            INTERMEDIATE_DATA_PATH.mkdir(parents=False, exist_ok=True)  # Makes the default intermediate data folder
        data_path = INTERMEDIATE_DATA_PATH

    # Make the Path to the Data File
    data_file_path = data_path / bird_id / session / data_name  # Note: data_name is the File Header

    if make_parents:
        # Check if the usual data hierarchical structure is there (creates it if it isn't)
        if not data_file_path.parents[0].exists():
            data_file_path.parents[0].mkdir(parents=True, exist_ok=True)

    data_file_path.resolve()
    assert data_file_path.exists(), f"{data_file_path} doesn't exist"

    return data_file_path


def _save_numpy_data(data: np.ndarray, data_name: str, bird_id: str, session: str, destination=None, make_parents=False):
    """

    :param data:
    :param data_name:
    :param bird_id:
    :param session:
    :param destination: Path(), optional
        Destination of data to be save other than the default intermediate location
    :param make_parents: bool, optional
        if True, it will create all of the parent folders for the Data File
    :return:
    """
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file

    data_file_path = _handle_data_path(data_name=data_name, bird_id=bird_id, session=session, dir_path=destination,
                                       make_parents=make_parents)

    print(f"Saving {data_name} Data to", data_file_path.name + '.npy')
    np.save(data_file_path, data)  # Save Data


def _load_numpy_data(data_name: str, bird_id: str, session: str, source=None):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file

    file_name = data_name + '.npy'

    data_file_path = _handle_data_path(data_name=file_name, bird_id=bird_id, session=session, dir_path=source,
                                       make_parents=False)

    print(f"Loading {data_name} Data from", data_file_path.name)

    return np.load(data_file_path)


def _save_pckl_data(data: np.ndarray, data_name: str, bird_id: str, session: str, destination=None, make_parents=False):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file

    file_name = data_name + '.pckl'  # Add the .pckl stem to the data_name

    data_file_path = _handle_data_path(data_name=file_name, bird_id=bird_id, session=session, dir_path=destination,
                                       make_parents=make_parents)  # Handle File Path and Directory Structure

    with open(data_file_path, "wb") as file_object:
        pickle.dump(data, file_object)

    print(f"Saving {data_name} Data to", data_file_path.name)


def _load_pckl_data(data_name: str, bird_id: str, session: str, source=None):
    file_name = data_name + '.pckl'

    data_file_path = _handle_data_path(data_name=file_name, bird_id=bird_id, session=session, dir_path=source,
                                       make_parents=False)

    # Safer Open procedure to make sure it isn't kept open
    with open(data_file_path, "rb") as file_object:
        data = pickle.load(file_object)

    print(f"Loading {data_name} Data from", data_file_path.name)

    return data


def _save_json_data(data: np.ndarray, data_name: str, bird_id: str, session: str, destination=None, make_parents=False):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file

    file_name = data_name + '.json'
    data_file_path = _handle_data_path(data_name=file_name, bird_id=bird_id, session=session, dir_path=destination,
                                       make_parents=make_parents)  # Handle File Path and Directory Structure

    with open(data_file_path, "w", encoding="utf8") as file_object:
        json.dump(data, file_object)

    print(f"Saving {data_name} Data to", data_file_path.name)


def _load_json_data(data_name: str, bird_id: str, session: str, source=None):
    file_name = data_name + '.json'
    data_file_path = _handle_data_path(data_name=file_name, bird_id=bird_id, session=session, dir_path=source,
                                       make_parents=False)

    # Safer Open procedure to make sure it isn't kept open
    with open(data_file_path, "r", encoding="utf8") as file_object:
        data = json.load(file_object)

    print(f"Loading {data_name} Data from", data_file_path.name)

    return data


