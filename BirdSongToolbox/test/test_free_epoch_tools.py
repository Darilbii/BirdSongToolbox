""" Test the free_epoch_tools.py module"""

import pytest
from BirdSongToolbox.file_utility_functions import _load_pckl_data
from BirdSongToolbox.free_epoch_tools import get_chunk_handlabels


@pytest.mark.run(order=1)
@pytest.fixture()
def bird_id():
    return 'z007'


@pytest.mark.run(order=1)
@pytest.fixture()
def session():
    return 'day-2016-09-09'


@pytest.mark.run(order=1)
@pytest.fixture()
def chunk_handlabels(bird_id, session, chunk_data_path):
    data_path = chunk_data_path
    return _load_pckl_data(data_name="chunk_handlabels_song", bird_id=bird_id, session=session,
                           source=data_path)

@pytest.mark.run(order=1)
def test_get_chunk_handlabels(chunk_handlabels):
    chunk_labels_list, chunk_onsets_list = get_chunk_handlabels(handlabels_list=chunk_handlabels)

    assert isinstance(chunk_labels_list, list)
    assert isinstance(chunk_onsets_list, list)
    assert len(chunk_onsets_list) == 2
    assert len(chunk_onsets_list[0]) == len(chunk_onsets_list[1]) == len(chunk_labels_list)




