""" Test the free_epoch_tools.py module"""

import pytest
from BirdSongToolbox.file_utility_functions import _load_pckl_data
from BirdSongToolbox.free_epoch_tools import get_chunk_handlabels, label_extractor


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
@pytest.fixture()
def onsets_testcase(chunk_handlabels):
    chunk_labels_list, chunk_onsets_list = get_chunk_handlabels(handlabels_list=chunk_handlabels)
    return chunk_onsets_list


@pytest.mark.run(order=1)
@pytest.fixture()
def labels_testcase(chunk_handlabels):
    chunk_labels_list, chunk_onsets_list = get_chunk_handlabels(handlabels_list=chunk_handlabels)
    return chunk_labels_list


@pytest.mark.run(order=1)
def test_get_chunk_handlabels(chunk_handlabels):
    chunk_labels_list, chunk_onsets_list = get_chunk_handlabels(handlabels_list=chunk_handlabels)

    assert isinstance(chunk_labels_list, list)
    assert isinstance(chunk_onsets_list, list)
    assert len(chunk_onsets_list) == 2
    assert len(chunk_onsets_list[0]) == len(chunk_onsets_list[1]) == len(chunk_labels_list)


@pytest.mark.run(order=1)
@pytest.mark.parametrize("test_instructions", [[1, 2, 3, 4], [1, [2, 3], 4], [1], [1, 'C', 'I'], ['BUFFER']])
def test_label_extractor(labels_testcase, onsets_testcase, test_instructions):

    specified_labels = label_extractor(all_labels=labels_testcase, starts=onsets_testcase[0],
                                       label_instructions=test_instructions)

    assert len(specified_labels) == len(test_instructions)



