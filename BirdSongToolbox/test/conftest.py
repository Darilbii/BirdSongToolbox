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