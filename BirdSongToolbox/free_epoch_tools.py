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