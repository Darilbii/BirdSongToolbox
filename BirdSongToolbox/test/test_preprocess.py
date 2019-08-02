import pytest
from BirdSongToolbox.file_utility_functions import _load_numpy_data, _load_pckl_data
from BirdSongToolbox.preprocess import common_average_reference_array, common_average_reference, bandpass_filter, \
    bandpass_filter_array_mne, bandpass_filter_array_ndsp, bandpass_filter_epochs, multi_bpf_epochs

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
def test_common_average_reference_array(chunk_neural_data):
    car_result = common_average_reference_array(neural_data=chunk_neural_data[0])

@pytest.mark.run(order=1)
def test_common_average_reference(chunk_neural_data):
    car_result = common_average_reference(chunk_neural=chunk_neural_data)


@pytest.mark.run(order=1)
def test_bandpass_filter_array_mne(chunk_neural_data):
    fs = 1000
    l_freq = 10
    h_freq = 20

    filt_data = bandpass_filter_array_mne(neural_data=chunk_neural_data[0], fs=fs, l_freq=l_freq, h_freq=h_freq)


@pytest.mark.run(order=1)
def test_bandpass_filter_array_ndsp(chunk_neural_data):
    fs = 1000
    l_freq = 10
    h_freq = 20
    filt_data = bandpass_filter_array_ndsp(neural_data=chunk_neural_data[0], fs=fs, l_freq=l_freq, h_freq=h_freq)

@pytest.mark.run(order=1)
def test_bandpass_filter(chunk_neural_data):
    fs = 1000
    l_freq = 10
    h_freq = 20

    filt_data = bandpass_filter(neural_data=chunk_neural_data[0], fs=fs, l_freq=l_freq, h_freq=h_freq)

@pytest.mark.run(order=1)
def test_bandpass_filter_epochs(chunk_neural_data):
    fs = 1000
    l_freq = 10
    h_freq = 20

    filt_data = bandpass_filter_epochs(epoch_neural_data=chunk_neural_data, fs=fs, l_freq=l_freq, h_freq=h_freq)


@pytest.mark.run(order=1)
def test_multi_bpf_epochs(chunk_neural_data):
    fs = 1000
    l_freqs = [10, 11, 12]
    h_freqs = [12, 13, 14]

    filt_data = multi_bpf_epochs(epoch_neural_data=chunk_neural_data, fs=fs, l_freqs=l_freqs, h_freqs=h_freqs,
                                 remove_edges=False, verbose=False)
