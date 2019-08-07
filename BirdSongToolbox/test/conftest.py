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
