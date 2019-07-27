from BirdSongToolbox.config.settings import DATA_DIR
from BirdSongToolbox.annotate.praat_utils import conv_textgrid_to_dict, textgrid_dict_to_handlabels_list, chunk_textgrid_dict_to_handlabels_dict
import numpy as np
import pytest
from praatio import tgio
from pathlib import Path


# TODO: Move Global Paths for Testing to a conftest.py file
@pytest.fixture(scope="module")
def praat_utils_path():
    data_path = DATA_DIR / "Chunk_TextGrids_Final"
    yield data_path

@pytest.fixture(scope="module")
def praat_utils_setup():
    data_path = DATA_DIR / "Chunk_TextGrids_Final"
    days_handlabels_dict = conv_textgrid_to_dict(bird_id='z007', session='day-2016-09-09', base_folder=data_path)
    yield days_handlabels_dict


@pytest.mark.run(order=1)
def test_conv_textgrid_to_dict():
    data_path = DATA_DIR / "Chunk_TextGrids_Final"
    days_handlabels_dict = conv_textgrid_to_dict(bird_id='z007', session='day-2016-09-09', base_folder=data_path)

    assert isinstance(days_handlabels_dict, dict)

@pytest.mark.run(order=1)
def test_chunk_textgrid_dict_to_handlabels_dict():
    data_path = DATA_DIR / "Chunk_TextGrids_Final"
    textgrid_dict = conv_textgrid_to_dict(bird_id='z007', session='day-2016-09-09', base_folder=data_path)
    handlabels_dict = chunk_textgrid_dict_to_handlabels_dict(textgrid_dict[0])

    assert isinstance(handlabels_dict, dict)
    assert 'labels' in handlabels_dict.keys()
    assert 'female' in handlabels_dict.keys()
    assert 'KWE' in handlabels_dict.keys()

    assert len(handlabels_dict['labels']) == 2
    assert isinstance(handlabels_dict['labels'], tuple)
    assert isinstance(handlabels_dict['labels'][0], list)
    assert isinstance(handlabels_dict['labels'][1], list)
    assert len(handlabels_dict['labels'][0]) == len(handlabels_dict['labels'][1][0]) == len(handlabels_dict['labels'][1][1])

    if len(handlabels_dict.keys())>3:
        assert 'old_epochs' in handlabels_dict.keys()

@pytest.mark.run(order=1)
def test_textgrid_dict_to_handlabels_list():
    data_path = DATA_DIR / "Chunk_TextGrids_Final"
    days_handlabels_dict = conv_textgrid_to_dict(bird_id='z007', session='day-2016-09-09', base_folder=data_path)

    handlabels = textgrid_dict_to_handlabels_list(days_handlabels_dict)

    assert isinstance(handlabels, list)