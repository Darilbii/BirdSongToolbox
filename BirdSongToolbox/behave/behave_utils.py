""" Utility Functions for Manipulating the Annotated Behavior """

from BirdSongToolbox.free_epoch_tools import get_event_related_1d

import numpy as np

def event_array_maker_1d(starts, ends, labels):
    """ Makes an array of the labels for one chunk

    Parameters
    ----------
    starts : list, shape [Epochs]->[Start Time]
        List of all Start Times corresponding to each event in the Chunk
    ends : list, shape [Epochs]->[Start Time]
        List of all End Times corresponding to each event in the Chunk
    labels : list, shape [Epochs]->[Labels]
        List of all Labels corresponding to each event in the Chunk

    Returns
    -------
    labels_array : array | shape (samples, 1)
        Array of the length of the chunk with each sample labeled based on the hand labeled events
    """

    abs_start = starts[0]
    abs_end = ends[-1]
    duration = abs_end - abs_start
    labels_array = np.zeros((int(duration / 30), 1))

    for start, end, label in zip(starts, ends, labels):
        if label == 'BUFFER':
            pass
        elif isinstance(label, int):
            labels_array[int(start / 30):int(end / 30)] = label
        elif isinstance(label, str):
            correction = {'I': 9, 'C': 10, 'X': 20}  # Convert Str Labels to the correct int value
            labels_array[int(start / 30):int(end / 30)] = correction[label]
        else:
            raise TypeError

    return labels_array


def event_array_maker_chunk(onsets_list, labels_list):
    """ Make an array of each of the Chunk's Labels

    Parameters
    ----------
    labels_list : list
        list of labels for all epochs for one day
        [Epoch] -> [Labels]
    onsets_list : list
        list of start and end times for all labels for one day
        [[Epochs]->[Start TIme] , [Epochs]->[End Time]]

    Returns
    -------
    chunk_labels_arrays : list | shape [Chunks]->(samples, 1)
        list of Arrays of the length of the each chunk with each sample labeled based on the hand labeled events
    """

    chunk_labels_arrays = []

    for starts, ends, labels in zip(onsets_list[0], onsets_list[1], labels_list):
        labels_array = event_array_maker_1d(starts=starts, ends=ends, labels=labels)
        chunk_labels_arrays.append(labels_array)

    return chunk_labels_arrays


def get_events_rasters(data, indices, fs, window, subtract_mean=None, **kwargs):
    """ Get behavior labels around all Instances of 1 Label from all Chunks (For Behavior Data)

    Parameters
    ----------
    data: list | shape [Chunks]->(Samples, 1)
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
        event_related_matrix = get_event_related_1d(data=chunk, fs=fs, indices=events, window=window,
                                                    subtract_mean=subtract_mean, **kwargs)

        if len(events) == 1:
            chunk_events.append(event_related_matrix)
        else:
            chunk_events.extend(event_related_matrix)

    chunk_events = np.asarray(chunk_events)

    return chunk_events


def repeat_events(labels_array):
    """ Repeat Instances for creating visualization of Behavior

    Parameters
    ----------
    labels_array : ndarray | (instances, samples)
        Array of the Events to be repeated for Visualization

    Returns
    -------
    set_array : ndarray | shape (~400, Samples)
        Array of the Events ready for visualization
    """

    set_width = 400

    num_inst, window_width = labels_array.shape  # Get the Shape of the Array
    num_repeats = int(set_width / num_inst)  # Caculate the Number of Repeats
    steps = np.arange(0, set_width, num_repeats)  # Calculate the Width of each Instance
    print(steps)

    print(steps[-1] + num_repeats)

    set_array = np.zeros((set_width, window_width))  # Create Empty Array
    print(np.shape(set_array))

    for inst, start in zip(labels_array, steps):
        set_array[start:start + num_repeats, :] = inst[None, :]

    if steps[num_inst] + num_repeats != set_width:
        extra = np.arange(steps[num_inst], set_width)
        set_array = np.delete(set_array, extra, axis=0)  # Trim Extra Rows

    return set_array

