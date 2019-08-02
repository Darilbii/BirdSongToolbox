""" Suite of Functions for Pre-Processing Chunked Neural Data """

import numpy as np

from neurodsp import filt
try:
    import mne
    # from mne.filter import filter_data
    _has_mne = True
except ImportError:
    _has_mne = False


def common_average_reference_array(neural_data, bad_channels: list = None):
    """ Applies a Common Average Reference to Neural Data

    Parameters
    ----------
    neural_data : array 2d, shape (Channels, Samples)
        Multi-Channel Neural Data
    bad_channels : list, optional
        list of Channels To Exclude from Common Average Reference

    Returns
    -------
    data_common_avg_ref : array 2d, shape (Channels, Samples)
        An array object of the Common Averaged Referenced Data
    """

    data_common_avg_ref = np.array(neural_data)

    # Exclude Noisy Channels from CAR if list of bad channels given
    channels_include = list(range(neural_data.shape[0]))

    if bad_channels is not None:
        channels_include = np.delete(channels_include, bad_channels)

    # Common Average Reference
    data_common_avg_ref = data_common_avg_ref - np.mean(data_common_avg_ref[channels_include, :], axis=0)[None, :]

    return data_common_avg_ref


def common_average_reference(chunk_neural, bad_channels: list = None):
    """ Common Average References all Epochs(Chunk)

    Paramters
    ---------
    chunk_neural : list, shape = [Chunk]->(Channels, Samples)
        Epoched(Chunk) Neural Data, list of 2darrays
    bad_channels : list, optional
        list of Channels To Exclude from Common Average Reference

    Returns
    -------
    car_chunk_neural : list, shape = [Chunk]->(Channels, Samples)
        list of Epoched(Chunks) Data that has been Common Averaged Referrenced
    """
    car_chunk_neural = []

    for chunk in chunk_neural:
        car_chunk_neural.append(common_average_reference_array(neural_data=chunk, bad_channels=bad_channels))
    return car_chunk_neural


def bandpass_filter_array_mne(neural_data, fs, l_freq: float, h_freq: float, fir_design='firwin2', verbose=False,
                              **kwargs):
    """ Bandpass Filters Neural Data using the MNE package

    Paramters
    ---------
    neural_data : ndarray, shape (…, n_times)
        The data to filter.
    sfreq : float
        The sample frequency in Hz.
    l_freq : float | None
        For FIR filters, the lower pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    h_freq : float | None
        For FIR filters, the upper pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    vebose : bool | False
        If True it will print out information about the filter used
    **kwargs : see MNE Documentation:
        [Github](https://github.com/mne-tools/mne-python/blob/d36440176cf3f3532f64e6f046c4a6a3eca028de/mne/filter.py#L742-L824)
        [mne](https://martinos.org/mne/stable/generated/mne.filter.filter_data.html)

    Returns
    -------
    out : ndarray, shape (…, n_times)
        The filtered data.
    """

    return mne.filter.filter_data(data=neural_data, sfreq=fs, l_freq=l_freq, h_freq=h_freq, fir_design=fir_design,
                                  verbose=verbose, **kwargs)


# NDSP (channels, samples)[base case]

def bandpass_filter_array_ndsp(neural_data, fs, l_freq: float, h_freq: float, remove_edges=False, **kwargs):
    """ Bandpass Filters Neural Data using the NeuroDSP package

    Parameters
    ----------
    neural_data : ndarray, shape (…, n_times)
        The data to filter. Defaults to working with either (epochs, channels, samples) or (channels, samples)
    fs : float
        The sample frequency in Hz.
    l_freq : float | None
        For FIR filters, the lower pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    h_freq : float | None
        For FIR filters, the upper pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    remove_edges : bool, optional, default: False
        If True, replace samples within half the kernel length to be np.nan.
        Only used for FIR filters.

    **kwargs : see NeuroDSP Documentation:
        [github](https://github.com/neurodsp-tools/neurodsp/blob/master/neurodsp/filt/filter.py#L13)
        [neurodsp](https://neurodsp-tools.github.io/neurodsp/generated/neurodsp.filt.filter_signal.html)

    Returns
    -------
    out: ndarray, shape (…, n_times)
        The filtered data.
    """

    # Apply the 1D Filtering Fuction accross the default Axis
    return np.apply_along_axis(func1d=filt.filter_signal, axis=-1, arr=neural_data, fs=fs, pass_type='bandpass',
                               f_range=(l_freq, h_freq), remove_edges=remove_edges, **kwargs)


def bandpass_filter(neural_data, fs, l_freq, h_freq, remove_edges=False, verbose=False, **kwargs):
    """ Bandpass Filter Neural Data

    Parameters:
    -----------
    neural_data : 2d array, shape (channels, samples)
        The Epoched Neural Data to be Bandpass Filtered
    fs : float
        The sample frequency in Hz.
    l_freq : float | None
        For FIR filters, the lower pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    h_freq : float | None
        For FIR filters, the upper pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    remove_edges : bool, optional, default: False
        If True, replace samples within half the kernel length to be np.nan.
        Only used for FIR filters when using neurodsp.
    vebose : bool, optional, default: False
        If True it will print out information about the filter used, (mne only)

    Returns
    -------
    filt_epochs : list, shape (channels, samples)
        The Neural Data Bandpass Filtered
    """

    # Switch for installed back-end for Filtering

    if _has_mne == True:
        filt_epochs = bandpass_filter_array_mne(neural_data=neural_data, fs=fs, l_freq=l_freq, h_freq=h_freq,
                                                verbose=verbose, **kwargs)
    else:
        filt_epochs = bandpass_filter_array_ndsp(neural_data=neural_data, fs=fs, l_freq=l_freq, h_freq=h_freq,
                                                 remove_edges=remove_edges, **kwargs)
    return filt_epochs


def bandpass_filter_epochs(epoch_neural_data, fs, l_freq, h_freq, remove_edges=False, verbose=False, **kwargs):
    """ Bandpass Filter Epochs(Chunks)

    Parameters:
    -----------
    epoch_neural_data : list, shape [Epoch]->(channels, samples)
        The Epoched Neural Data to be Bandpass Filtered
    fs : float
        The sample frequency in Hz.
    l_freq : float | None
        For FIR filters, the lower pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    h_freq : float | None
        For FIR filters, the upper pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    remove_edges : bool, optional, default: False
        If True, replace samples within half the kernel length to be np.nan.
        Only used for FIR filters when using neurodsp.
    vebose : bool, optional, default: False
        If True it will print out information about the filter used, (mne only)

    Returns
    -------
    filt_epochs : list, shape [Epoch]->(channels, samples)
        The Epoched Neural Data Bandpass Filtered
    """

    filt_epochs = []
    for epoch in epoch_neural_data:
        filt_epochs.append(bandpass_filter(neural_data=epoch, fs=fs, l_freq=l_freq, h_freq=h_freq,
                                           remove_edges=remove_edges, verbose=verbose, **kwargs))
    return filt_epochs


def multi_bpf_epochs(epoch_neural_data, fs, l_freqs, h_freqs, remove_edges=False, verbose=False, **kwargs):
    """

    Parameters:
    -----------
    epoch_neural_data : list, shape [Epoch]->(channels, samples)
        The Epoched Neural Data to be Bandpass Filtered
    fs : float
        The sample frequency in Hz.
    l_freqs : array-like | None
        For FIR filters, the lower pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    h_freqs : array-like | None
        For FIR filters, the upper pass-band edge; for IIR filters, the upper cutoff frequency.
        If None the data are only low-passed.
    remove_edges : bool, optional, default: False
        If True, replace samples within half the kernel length to be np.nan.
        Only used for FIR filters when using neurodsp.
    vebose : bool, optional, default: False
        If True it will print out information about the filter used, (mne only)

    Returns
    -------
    multi_filt_epochs : list, shape [Freq]->[Epoch]->(channels, samples)
        The Epoched Neural Data Bandpass Filtered

    """

    assert len(l_freqs) == len(
        h_freqs), 'l_freqs and h_freqs must be the same length, {l_f} not equal to {h_f}'.format(l_f=len(l_freqs),
                                                                                                 h_f=len(h_freqs))
        # f'l_freqs and h_freqs must be the same length, {len(l_freqs)} not equal to {len(h_freqs)}'

    multi_filt_epochs = []
    for l_freq, h_freq in zip(l_freqs, h_freqs):
        multi_filt_epochs.append(bandpass_filter_epochs(epoch_neural_data=epoch_neural_data, fs=fs, l_freq=l_freq,
                                                        h_freq=h_freq, remove_edges=remove_edges, verbose=verbose,
                                                        **kwargs))
    return multi_filt_epochs

