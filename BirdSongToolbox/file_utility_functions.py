from BirdSongToolbox.config.settings import INTERMEDIATE_DATA_PATH

import numpy as np
import json
import pickle


def _save_numpy_data(data: np.ndarray, data_name: str, bird_id: str, session: str):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file
    # Save Song Data
    if not INTERMEDIATE_DATA_PATH.exists():
        INTERMEDIATE_DATA_PATH.mkdir(parents=False, exist_ok=True)

    # file_name = data_name + '_' + bird_id + '_' + session

    # data_file_path = INTERMEDIATE_DATA_PATH / file_name

    data_file_path = INTERMEDIATE_DATA_PATH / bird_id / session / data_name

    # Check if the usual data hierachical structure is there (creates it if it isn't)
    if not data_file_path.parents[0].exists():
        data_file_path.parents[0].mkdir(parents=True, exist_ok=True)

    print(f"Saving {data_name} Data to", data_file_path.name + '.npy')
    np.save(data_file_path, data)


def _load_numpy_data(data_name: str, bird_id: str, session: str):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file

    file_name = data_name + '.npy'

    data_file_path = INTERMEDIATE_DATA_PATH / bird_id / session / file_name

    print(f"Loading {data_name} Data from", data_file_path.name)

    return np.load(data_file_path)


def _save_pckl_data(data: np.ndarray, data_name: str, bird_id: str, session: str):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file
    # Save Song Data
    if not INTERMEDIATE_DATA_PATH.exists():
        INTERMEDIATE_DATA_PATH.mkdir(parents=False, exist_ok=True)

    file_name = data_name + '.pckl'
    data_file_path = INTERMEDIATE_DATA_PATH / bird_id / session / file_name

    # Check if the usual data hierachical structure is there (creates it if it isn't)
    if not data_file_path.parents[0].exists():
        data_file_path.parents[0].mkdir(parents=True, exist_ok=True)

    with open(data_file_path, "wb") as file_object:
        json.dump(data, file_object)

    print(f"Saving {data_name} Data to", data_file_path.name)


def _load_pckl_data(data_name: str, bird_id: str, session: str):

    file_name = data_name + '.pckl'

    data_file_path = INTERMEDIATE_DATA_PATH / bird_id / session / file_name

    # Safer Open procedure to make sure it isn't kept open
    with open(data_file_path, "rb") as file_object:
        data = pickle.load(file_object)

    print(f"Loading {data_name} Data from", data_file_path.name)

    return data


def _save_json_data(data: np.ndarray, data_name: str, bird_id: str, session: str):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file
    # Save Song Data
    if not INTERMEDIATE_DATA_PATH.exists():
        INTERMEDIATE_DATA_PATH.mkdir(parents=False, exist_ok=True)

    # file_name = data_name + '_' + bird_id + '_' + session

    # data_file_path = INTERMEDIATE_DATA_PATH / file_name
    file_name = data_name + '.json'
    data_file_path = INTERMEDIATE_DATA_PATH / bird_id / session / file_name

    # Check if the usual data hierachical structure is there (creates it if it isn't)
    if not data_file_path.parents[0].exists():
        data_file_path.parents[0].mkdir(parents=True, exist_ok=True)

    with open(data_file_path, "w", encoding="utf8") as file_object:
        json.dump(data, file_object)

    print(f"Saving {data_name} Data to", data_file_path.name)


def _load_json_data(data_name: str, bird_id: str, session: str):

    file_name = data_name + '.json'
    data_file_path = INTERMEDIATE_DATA_PATH / bird_id / session / file_name

    # Safer Open procedure to make sure it isn't kept open
    with open(data_file_path, "r", encoding="utf8") as file_object:
        data = json.load(file_object)

    print(f"Loading {data_name} Data from", data_file_path.name)

    return data
