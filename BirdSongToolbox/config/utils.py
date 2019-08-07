"""Utility Functions for Handling the User Designated Paths"""

import yaml
import warnings
from pathlib import Path

FILE_PATH = Path(__file__)
CONFIG_PATH = FILE_PATH.parent / "config.yaml"  # Location of the Config File


def _make_default_config_file():
    default = {"Chunked_Data_Path": "", "PrePd_Data_Path": "", "Raw_Data_Path": "", "Intermediate_Path": "",
               "User_Defined_Paths": {}}

    _save_config(default)


def _save_config(configuration: dict):
    """ Saves a Dictionary of Paths to the config.yaml"""

    # Make Sure the Configuration is formatted Correctly
    assert "Chunked_Data_Path" in configuration.keys()
    assert "PrePd_Data_Path" in configuration.keys()
    assert "Raw_Data_Path" in configuration.keys()
    assert "Intermediate_Path" in configuration.keys()
    assert "User_Defined_Paths" in configuration.keys()

    with open(str(CONFIG_PATH), "w") as file_object:
        yaml.dump(configuration, file_object)


def _load_config():
    """ Imports the Configuration as a dict"""

    with open(str(CONFIG_PATH), "r") as file_object:
        configuration = yaml.safe_load(file_object)
    return configuration


def get_spec_config_path(specific_path: str):
    """ Get the Specified Data Path"""
    try:
        configuration = _load_config()  # Load the Configuration as dict

    except FileNotFoundError:
        _make_default_config_file()  # Make the Config File if it doesn't already exist
        configuration = _load_config()  # Load the Configuration as dict
        warnings.warn('There is currently No default Paths for BirdSongToolbox. Please update the configuration if '
                      'you want to use the default paths convenience api of BirdSongToolbox')

    return Path(configuration[specific_path])


def update_config_path(specific_path: str, new_path):
    """ Update the default data paths in the Configuration Yaml file

    Parameters
    ----------
    specific_path : str
        Specific Default Path to be updated
    new_path : str, Path
        New User given Path to be used by Default by BirdSongToolbox
    """

    default_keys = ["Chunked_Data_Path", "PrePd_Data_Path", "Raw_Data_Path", "Intermediate_Path", "User_Defined_Paths"]

    assert specific_path in default_keys, "{path} is not a path in the default path".format(path=specific_path)

    if isinstance(new_path, Path):
        assert new_path.exists()
    elif isinstance(new_path, str):
        new_path = Path(new_path)
        assert new_path.exists()
    else:
        raise TypeError

    current_configuration = _load_config()

    if specific_path == "User_Defined_Paths":
        current_configuration["User_Defined_Paths"][specific_path] = str(new_path)  # Add to the nested dictionary
    else:
        current_configuration[specific_path] = str(new_path)  # Update the Path in the root dictionary

    _save_config(configuration=current_configuration)


def get_config_from_yaml(source: str or Path):

    with open(str(source), "r") as file_object:
        configuration = yaml.safe_load(file_object)

    _save_config(configuration=configuration)
