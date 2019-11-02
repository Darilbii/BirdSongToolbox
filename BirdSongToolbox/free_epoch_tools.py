""" Functions for Epoching Free Behaviar into more Traditional Trialized Data Formats"""

import numpy as np


def get_chunk_handlabels(handlabels_list):
    """ Get all of the Hand-labels from all of the 'labels' tiers from the handlabels_list

    Parameters
    ----------
    handlabels_list : list
        List of all Dictionaries containing all of the handlabels for each chunk for one day
        shape: [chunk_num] -> {'labels' : [labels, onsets],
                               'female' : [labels, onsets],
                               'KWE' : [time],
                               'old_epochs' : {'epoch_#' : [labels, onsets]} # Only if Labeled the old way
                               }

    Returns
    -------
    labels_list : list
        list of labels for all epochs for one day
        [Epoch] -> [Labels]
    onsets_list : list
        list of start and end times for all labels for one day
        [[Epochs]->[Start TIme] , [Epochs]->[End Time]]
    """
    labels_list = []
    starts_list = []
    ends_list = []

    for chunk in handlabels_list:
        labels, [starts, ends] = chunk['labels']  # Get Label Compnents
        labels_list.append(labels)
        starts_list.append(starts)
        ends_list.append(ends)

    onsets_list = [starts_list, ends_list]

    return labels_list, onsets_list


def label_focus_chunk(focus, labels, starts):
    """ Create a list of every instance of the User defined User Label (Focus on One Label)

    Parameters
    ----------
    focus : str or int
        User defined Label to focus on
    labels : list
        List of all Labels corresponding to each Epoch in Full_Trials
        [Epochs]->[Labels]
    starts : list
        List of all Start Times corresponding to each Epoch in Full_Trials
        [Epochs]->[Start Time]

    Returns
    -------
    label_index : list
        List of all start frames of every instances of the label of focus
        [Num_Trials]->[Num_Exs]
    """
    label_index = []

    # for i in range(len(labels)):
    #     Trial_Labels = [int(starts[i][x] / 30) for x in range(len(labels[i])) if labels[i][x] == Focus]

    for start, epoch in zip(starts, labels):
        trial_labels = [start[i] for i, x in enumerate(epoch) if x == focus]
        label_index.append(trial_labels)
    return label_index


# Function for Grouping Multiple Labels into 1 Label (e.g. Combine Calls and Introductory Notes)

def label_group_chunk(Focuses, labels, starts):
    """Group Selected labels together into One Label e.g. Combine Calls and Intro. Notes (Group these labels together)

    Parameters
    ----------
    focus : str or int
        User defined Label to focus on
    labels : list, shape [Epochs]->[labels]
        List of all labels corresponding to each Epoch in Full_Trials
    starts : list, shape [Epochs]->[Start Time]
        List of all Start Times corresponding to each Epoch in Full_Trials

    Returns
    -------
    label_index : list, shape [Num_Trials]->[Num_Exs]
        List of all start frames of every instances of the label of focus
    """

    label_index = []

    for start, chunk in zip(starts, labels):
        trial_labels = [start[index] for index, label in enumerate(chunk) if label in Focuses]
        label_index.append(trial_labels)
    return label_index


# Function for grabing more examples from a onset


def label_extractor(all_labels, starts, label_instructions):
    """Extracts all of the Neural Data Examples of User Selected Labels and return them in the designated manner.

    Label_Instructions = tells the Function what labels to extract and whether to group them together

    Parameters
    ----------
    all_labels : list, shape [Epochs]->[Labels]
        List of all Labels corresponding to each Epoch in Full_Trials
    starts : list, shape [Epochs]->[Start Time]
        List of all Start Times corresponding to each Epoch in Full_Trials
    label_instructions : list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label

    Returns
    -------
    specified_labels : list, shape [Labels]->[Chunk]->[Times]
        List containing the instances of the Labels for each Chunk

    """

    specified_labels = []

    for instruction in range(len(label_instructions)):
        if type(label_instructions[instruction]) == int or type(label_instructions[instruction]) == str:
            label_starts = label_focus_chunk(label_instructions[instruction], all_labels, starts)
        else:
            label_starts = label_group_chunk(label_instructions[instruction], all_labels, starts)
        specified_labels.append(label_starts)

    # if len(specified_labels) == 1:
    #     specified_labels = specified_labels[0]

    return specified_labels


def get_event_related_1d(data, fs, indices, window, subtract_mean=None, overlapping=None, **kwargs):
    """Take an input time series, vector of event indices, and window sizes,
        and return a 2d matrix of windowed trials around the event indices.

        Parameters
        ----------
        data : array-like 1d
            Voltage time series
        fs : int
            Sampling Frequency
        indices : array-like 1d of integers
            Indices of event onset indices
        window : tuple | shape (start, end)
            Window (in ms) around event onsets, window components must be integer values
        subtract_mean : tuple, optional | shape (start, end)
            if present, subtract the mean value in the subtract_mean window for each
            trial from that trial's time series (this is a trial-by-trial baseline)
        overlapping : list, optional
            Samples that overlap with another Epoch (overlapping_samples - epoch_abs_start)

        Returns
        -------
        event_related_matrix : array-like 2d
            Event-related times series around each index
            Each row is a separate event
        """

    def windows_to_indices(fs, window_times):
        """convert times (in ms) to indices of points along the array"""
        conversion_factor = (1 / fs) * 1000  # convert from time points to ms
        window_times = np.floor(np.asarray(window_times) / conversion_factor)  # convert
        window_times = window_times.astype(int)  # turn to ints

        return window_times

    def convert_index(fs, indexes):
        """convert the start times to their relative sample based on the fs parameter"""
        conversion_factor = (1 / fs) * 30000  # Convert from 30Khs to the set sampling rate
        indexes = np.rint(np.array(indexes) / conversion_factor)
        indexes = indexes.astype(int)
        return indexes

    # Remove overlapping labels
    if overlapping is not None:
        overlaps = [index for index, value in enumerate(indices) if value in overlapping]  # Find overlapping events
        indices = np.delete(indices, overlaps, axis=0)  # Remove them from the indices

    window_idx = windows_to_indices(fs=fs, window_times=window)  # convert times (in ms) to indices
    inds = convert_index(fs=fs, indexes=indices) + np.arange(window_idx[0], window_idx[1])[:, None]  # broadcast indices

    # Remove Edge Instances from the inds
    bad_label = []
    bad_label.extend([index for index, value in enumerate(inds[0, :]) if value < 0])  # inds that Start before Epoch
    bad_label.extend([index for index, value in enumerate(inds[-1, :]) if value >= len(data)])  # inds End after Epoch
    inds = np.delete(inds, bad_label, axis=1)  # Remove Edge Instances from the inds

    event_times = np.arange(window[0], window[1], (1 / fs) * 1000)
    event_related_matrix = data[inds]  # grab the data
    event_related_matrix = np.squeeze(event_related_matrix).T  # make sure it's in the right format

    # baseline, if requested
    if subtract_mean is not None:
        basewin = [0, 0]
        basewin[0] = np.argmin(np.abs(event_times - subtract_mean[0]))
        basewin[1] = np.argmin(np.abs(event_times - subtract_mean[1]))
        event_related_matrix = event_related_matrix - event_related_matrix[:, basewin[0]:basewin[1]].mean(axis=1,
                                                                                                          keepdims=True)

    return event_related_matrix


# TODO: Revisit the Function Below and its current Utility
def make_event_times_axis(window, fs):
    """
    Parameters
    ----------
    window : tuple (integers)
        Window (in ms) around event onsets
    fs : int
        Sampling Frequency

    Returns
    -------
    event_times : array
        Array of the Times Indicated in ms
    """
    event_times = np.arange(window[0], window[1], (1 / fs) * 1000)
    return event_times


def get_event_related_2d(data, indices, fs, window, subtract_mean=None, overlapping=None, **kwargs):
    """
    Parameters
    ----------
    data: list, shape (Channels, Samples)
        Neural Data
    indices : list, shape [Events]
        Onsets of the Labels to be Clipped for one Chunk
    fs : int
            Sampling Frequency
    indices : array-like 1d of integers
            Indices of event onset indices
    window : tuple | shape (start, end)
        Window (in ms) around event onsets, window components must be integer values
    subtract_mean : tuple, optional | shape (start, end)
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)
    overlapping : list, optional
        Samples that overlap with another Epoch (overlapping_samples - epoch_abs_start)

    Returns
    -------
    events_matrix : ndarray | shape (Instances, Channels, Samples)
        Neural Data in the User Defined Window for all Instances of each Label
    """

    all_channel_events = np.apply_along_axis(func1d=get_event_related_1d, axis=-1, arr=data, fs=fs, indices=indices,
                                             window=window, subtract_mean=subtract_mean, overlapping=overlapping,
                                             **kwargs)
    events_matrix = np.transpose(all_channel_events, axes=[1, 0, 2])  # Reshape to (Events, Ch, Samples)
    return events_matrix


def get_event_related_nd(data, indices, fs, window, subtract_mean=None, overlapping=None, **kwargs):
    """ Take an input ndarray of time series data, vector of event indices, and window sizes, and return a nd matrix
    of windowed trials around the event indices.

    Parameters
    ----------
    data: list, shape (Channels, Samples) or (Frequencies, Channels, Samples)
        Neural Data
    indices : list, shape [Events]
        Onsets of the Labels to be Clipped for one Chunk
    fs : int
            Sampling Frequency
    indices : array-like 1d of integers
            Indices of event onset indices
    window : tuple | shape (start, end)
        Window (in ms) around event onsets, window components must be integer values
    subtract_mean : tuple, optional | shape (start, end)
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)
    overlapping : list, optional
        Samples that overlap with another Epoch (overlapping_samples - epoch_abs_start)

    Returns
    -------
    events_matrix : ndarray | shape (Instances, Channels, Samples) or (Instances, Frequencies, Channels, Samples)
        Neural Data in the User Defined Window for all Instances of each Label
    """

    all_channel_events = np.apply_along_axis(func1d=get_event_related_1d, axis=-1, arr=data, fs=fs, indices=indices,
                                             window=window, subtract_mean=subtract_mean, overlapping=overlapping,
                                             **kwargs)
    if len(all_channel_events.shape) < 4:
        events_matrix = all_channel_events  # Reshape to (Frequencies, Ch, Samples)

    else:
        events_matrix = np.transpose(all_channel_events,
                                     axes=[2, 0, 1, 3])  # Reshape to (Events, Frequencies, Ch, Samples)

    return events_matrix


def get_event_related(data, indices, fs, window, subtract_mean=None, **kwargs):
    """ Get all Instances of 1 Label from all Chunks

    Parameters
    ----------
    data: list | shape [Chunks]->(Channels, Samples)
        Neural Data
    indices : list | shape [Chunks]->[Events]
        Onsets of the Labels to be Clipped
    fs : int
        Sampling Frequency
    indices : array-like 1d of integers
        Indices of event onset indices
    window : tuple | shape (start, end)
        Window (in ms) around event onsets, window components must be integer values
    subtract_mean : tuple, optional | shape (start, end)
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)

    Returns
    -------
    events_matrix : ndarray | shape (Instances, Channels, Samples)
        Neural Data in the User Defined Window for all Instances of each Label
    """

    chunk_events = []

    for chunk, events in zip(data, indices):
        event_related_matrix = get_event_related_2d(data=chunk, fs=fs, indices=events, window=window,
                                                    subtract_mean=subtract_mean, **kwargs)
        chunk_events.extend(event_related_matrix)

    chunk_events = np.asarray(chunk_events)

    return chunk_events


def event_clipper(data, label_events, fs, window, subtract_mean=None, **kwargs):
    """Get all of the Instances for all Labels given for one set of chunks

    Parameters
    ----------
    data: list | shape [Chunks]->(Channels, Samples)
        Neural Data
    label_events : list, shape [Label]->[Chunks]->[Events]
        Onsets of the Labels to be Clipped
    fs : int
        Sampling Frequency
    window : tuple | shape (start, end)
        Window (in ms) around event onsets, window components must be integer values
    subtract_mean : tuple, optional | shape (start, end)
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)

    Returns
    -------
    chunk_events : list | shape [Labels]->(Instances, Channels, Samples)
        Neural Data in the User Defined Window for all Instances of each Label

    """

    chunk_events = []

    for index, events in enumerate(label_events):
        event_related_matrix = get_event_related(data=data, indices=events, fs=fs, window=window,
                                                 subtract_mean=subtract_mean, **kwargs)
        chunk_events.append(event_related_matrix)  # Append to List

    return chunk_events


def event_clipper_freqs(filt_data, label_events, fs, window, subtract_mean=None, **kwargs):
    """ Get all of the Instances for all Labels given for all frequency bands for one set of chunks

    Parameters
    ----------
    filt_data: list | shape [Freq]->[Chunks]->(Channels, Samples)
        Neural Data
    label_events : list | shape [Label]->[Chunks]->[Events]
        Onsets of the Labels to be Clipped
    fs : int
        Sampling Frequency
    window : tuple | shape (start, end)
        Window (in ms) around event onsets, window components must be integer values
    subtract_mean : tuple, optional | shape (start, end)
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)

    Returns
    -------
    chunk_events : list | shape [Labels]->[Freq]->(Instances, Channels, Samples)
        Neural Data in the User Defined Window for all Instances of each Label
    """

    chunk_events = []

    for label in label_events:
        freq_events = []
        for freq in filt_data:
            event_matrix = get_event_related(data=freq, indices=label, fs=fs, window=window,
                                             subtract_mean=subtract_mean, **kwargs)
            freq_events.append(event_matrix)
        chunk_events.append(freq_events)

    return chunk_events


def get_event_related_nd_chunk(chunk_data, chunk_indices, fs, window, subtract_mean=None, overlapping=None, **kwargs):
    """ Run the get_event_related_nd across all chunks

    Parameters
    ----------
    chunk_data: list | shape [Chunks]->(Channels, Samples) or (Frequencies, Channels, Samples)
        Neural Data
    chunk_indices : list, shape [Chunks]->[Events]
        Onsets of the Labels to be Clipped
    fs : int
        Sampling Frequency
    window : tuple | shape (start, end)
        Window (in ms) around event onsets, window components must be integer values
    subtract_mean : tuple, optional | shape (start, end)
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)

    Returns
    -------
    chunk_instances : ndarray | shape (Instances, Channels, Samples) or (Instances, Frequencies, Channels, Samples)
        event related data
    """
    chunk_instances = []

    for data, indices, in zip(chunk_data, chunk_indices):
        # TODO: Check if this needs to account for chunks that have no instances

        instances = get_event_related_nd(data=data, indices=indices, fs=fs, window=window,
                                         subtract_mean=subtract_mean, overlapping=overlapping, **kwargs)

        if len(instances) == 1:
            chunk_instances.extend(instances)
        else:
            chunk_instances.append(instances)

    return chunk_instances

def event_clipper_nd(data, label_events, fs, window, subtract_mean=None, **kwargs):
    """Get all of the Instances for all Labels given for one set of chunks

    Parameters
    ----------
    data: list | shape [Chunks]->(Channels, Samples) or (Freqs, Channels, Samples)
        Neural Data
    label_events : list, shape [Label]->[Chunks]->[Events]
        Onsets of the Labels to be Clipped
    fs : int
        Sampling Frequency
    window : tuple | shape (start, end)
        Window (in ms) around event onsets, window components must be integer values
    subtract_mean : tuple, optional | shape (start, end)
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)

    Returns
    -------
    chunk_events : list | shape [Labels]->(Instances, Channels, Samples) or (Instances, Frequencies, Channels, Samples)
        Neural Data in the User Defined Window for all Instances of each Label

    """

    chunk_events = []

    for events in label_events:
        event_related_matrix = get_event_related_nd_chunk(chunk_data=data, chunk_indices=events, fs=fs,
                                                          window=window, subtract_mean=subtract_mean, **kwargs)

        corr_shape = event_shape_correction(event_related_matrix)  # Format shape of list to be ndarray indexible

        chunk_events.append(corr_shape)  # Append to List

    return chunk_events


def event_shape_correction(chunk_events):
    """ Reshape the output of get_event_related_nd_chunk to be shape of [Instances]->( Freqs, Channels, Samples)"""
    corrected = []
    for chunk in chunk_events:
        if len(chunk.shape) == 4:
            for instances in chunk:
                corrected.append(instances)
        else:
            corrected.append(chunk)
    return corrected

def long_silence_finder(silence, all_labels, all_starts, all_ends, window):
    """ Checks if the Duration of the Silence Label is longer than the window and sets start equal to the middle of event

    Parameters
    ----------
    silence : str or int
        User defined Label to focus on
    all_labels : list
        List of all Labels corresponding to each Chunk in Full_Trials
        [Epochs]->[Labels]
    all_starts : list
        List of all Start Times corresponding to each Chunk in Full_Trials
        [Epochs]->[Start Time]
    all_ends : list
        List of all End Times corresponding to each Chunk in Full_Trials
        [Epochs]->[End Time]
    window : tuple | shape (start, end)
            Window (in ms) around event onsets, window components must be integer values

    Returns
    -------
    label_index : list
        List of all start frames of every instances of the label of focus
        [Num_Trials]->[Num_Exs]
    """

    label_index = []
    fs = 30  # Originally Sammpling is 30Khz

    window_len = len(np.arange(window[0], window[1]))* fs  # Length of the Window

    for starts, ends, labels in zip(all_starts, all_ends, all_labels):
        mid_starts = [start + ((end - start) / 2) for start, end, label in zip(starts, ends, labels) if
                      label == silence and (end - start) > window_len]
        label_index.append(mid_starts)
    return label_index
