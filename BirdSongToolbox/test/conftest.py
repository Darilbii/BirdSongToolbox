"""Configuration file for pytest for BirdSongToolbox."""

import pytest

from pathlib import Path

###################################################################################################
###################################################################################################

# I guess I'll eventually add something here
@pytest.mark.run(order=0)
@pytest.fixture(scope="module")
def data_path():
    # Test Data Paths
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    return data_dir


@pytest.mark.run(order=0)
@pytest.fixture(scope="module")
def chunk_data_path(data_path):
    test_data_chunk_data_demo_dir = data_path / "Chunk_Data_Demo"
    return test_data_chunk_data_demo_dir


@pytest.mark.run(order=0)
@pytest.fixture(scope="module")
def PrePd_data_dir_path(data_path):
    """ Directory of Test Data of the PrePd Data Path"""
    path = data_path / "PrePd_Data"
    return path

@pytest.mark.run(order=0)
@pytest.fixture(scope="module")
def data_pipeline_class_pckl_path(data_path):
    """ Directory of the Test Data of a Pipeline() Class Instance"""
    path = data_path / "Pipeline_Test_Dataz020_day-2016-06-02.pckl"
    return path

@pytest.mark.run(order=0)
@pytest.fixture(scope="module")
def data_praat_utils_dir_path(data_path):
    path = data_path / "Chunk_TextGrids_Final"
    return path

# @pytest.mark.run(order=0)
# @pytest.fixture(scope="module")
# def chunk_data_path_name():
#     return "Chunked_Data_Path"
#
# @pytest.mark.run(order=0)
# @pytest.fixture(scope="module")
# def create_test_config(chunk_data_path_name, chunk_data_path ):
#     from BirdSongToolbox.config.utils import update_config_path
#     from BirdSongToolbox.config.utils import _make_default_config_file
#     from BirdSongToolbox.config.utils import CONFIG_PATH
#
#     _make_default_config_file()  # Make default Config File
#     update_config_path(specific_path=chunk_data_path_name, new_path=chunk_data_path)  # Make Default path for Chunk Data
#
#     yield
#
#     CONFIG_PATH.unlink()  # Remove the dummy config file
