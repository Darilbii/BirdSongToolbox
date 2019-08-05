""" Functions for Epoching Free Behaviar into more Traditional Trialized Data Formats"""


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
