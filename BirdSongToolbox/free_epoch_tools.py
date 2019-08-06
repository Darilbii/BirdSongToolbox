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
        data : float
            Data sampling rate (Hz)
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
    inds = convert_index(fs=fs, indexes=indices) + np.arange(window_idx[0],
                                                             window_idx[1])[:, None]  # build matrix of indices

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


def get_event_related(data, indices, fs, window, subtract_mean=None, overlapping=None, **kwargs):
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
    overlapping : list
        Samples that overlap with another Epoch (overlapping_samples - epoch_abs_start)

    Returns
    -------
    events_matrix : ndarray, shape (Instances, Channels, Samples)
        Neural Data in the User Defined Window for all Instances of each Label
    """

    all_channel_events = np.apply_along_axis(func1d=get_event_related_1d, axis=-1, arr=data, fs=fs, indices=indices,
                                             window=window, subtract_mean=subtract_mean, overlapping=overlapping,
                                             **kwargs)
    events_matrix = np.transpose(all_channel_events, axes=[1, 0, 2])  # Reshape to (Events, Ch, Samples)
    return events_matrix


