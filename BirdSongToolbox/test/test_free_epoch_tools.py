""" Test the free_epoch_tools.py module"""

import pytest
import numpy as np

from BirdSongToolbox.file_utility_functions import _load_pckl_data
from BirdSongToolbox.preprocess import multi_bpf_epochs
from BirdSongToolbox.free_epoch_tools import get_chunk_handlabels, label_extractor, get_event_related_1d, \
    get_event_related_2d, get_event_related, event_clipper, event_clipper_freqs


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
def chunk_neural_data(bird_id, session, chunk_data_path):
    data_path = chunk_data_path
    return _load_pckl_data(data_name="Large_Epochs_Neural_Song", bird_id=bird_id, session=session,
                           source=data_path)


@pytest.mark.run(order=1)
@pytest.fixture()
def chunk_handlabels(bird_id, session, chunk_data_path):
    data_path = chunk_data_path
    return _load_pckl_data(data_name="chunk_handlabels_Song", bird_id=bird_id, session=session,
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
@pytest.fixture()
def starts_1chunk(labels_testcase, onsets_testcase):
    test_instructions = [1]
    specified_labels = label_extractor(all_labels=labels_testcase, starts=onsets_testcase[0],
                                       label_instructions=test_instructions)

    return specified_labels[0][0]


@pytest.mark.run(order=1)
@pytest.fixture()
def starts_multi_chunk(labels_testcase, onsets_testcase):
    test_instructions = [1]
    specified_labels = label_extractor(all_labels=labels_testcase, starts=onsets_testcase[0],
                                       label_instructions=test_instructions)

    return specified_labels[0]


@pytest.mark.run(order=1)
@pytest.fixture()
def multi_starts(labels_testcase, onsets_testcase):
    test_instructions = [1, 2, 3, 4]
    specified_labels = label_extractor(all_labels=labels_testcase, starts=onsets_testcase[0],
                                       label_instructions=test_instructions)

    return specified_labels


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


@pytest.mark.run(order=1)
@pytest.mark.parametrize("windows,subtract,overlap", [((-50, 50), None, None),
                                                      ((-100, 100), (-100, 0), None),
                                                      ((0, 100), None, None),
                                                      ((-100, 0), None, None)])
def test_get_event_related_1d(chunk_neural_data, starts_1chunk, windows, subtract, overlap):
    # TODO: Write Parameterize functions to test out the Overlap for the Old Epochs

    data_1d = chunk_neural_data[0][0, :]

    test_related_matrix = get_event_related_1d(data=data_1d, fs=1000, indices=starts_1chunk,
                                               window=windows, subtract_mean=subtract, overlapping=overlap)


@pytest.mark.run(order=1)
def test_get_event_related_2d(chunk_neural_data, starts_1chunk):
    data_2d = chunk_neural_data[0]  # Grab one Chunk (channels, samples)

    events_matrix = get_event_related_2d(data=data_2d, indices=starts_1chunk, fs=1000, window=(-50, 50))

    num_instances, num_chans, samples = np.shape(events_matrix)

    assert len(starts_1chunk) == num_instances
    assert len(data_2d) == num_chans
    assert samples == 100  # Bad Test, but a start


@pytest.mark.run(order=1)
def test_get_event_related(chunk_neural_data, starts_multi_chunk):
    events_matrix = get_event_related(data=chunk_neural_data, indices=starts_multi_chunk, fs=1000, window=(-50, 50))


@pytest.mark.run(order=1)
def test_event_clipper(chunk_neural_data, multi_starts):
    events_matrix = event_clipper(data=chunk_neural_data, label_events=multi_starts, fs=1000, window=(-50, 50))
    assert len(events_matrix) == len(multi_starts)


@pytest.mark.run(order=1)
@pytest.mark.parametrize("lows,highs", [([10, 30], [20, 40]), ([10, 40, 100], [30, 70, 150])])
def test_event_clipper_freqs(chunk_neural_data, multi_starts, lows, highs):
    filt_data = multi_bpf_epochs(epoch_neural_data=chunk_neural_data, fs=1000, l_freqs=lows, h_freqs=highs)
    events_matrix = event_clipper_freqs(filt_data=filt_data, label_events=multi_starts, fs=1000, window=(-50, 50))

    assert len(events_matrix) == len(multi_starts)
    assert len(events_matrix[0]) == len(lows) == len(highs) == len(filt_data)
