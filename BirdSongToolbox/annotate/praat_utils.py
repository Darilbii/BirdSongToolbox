""" Utility functions for Behavioral Handlabelling using Praat"""

import numpy as np
from praatio import tgio
from math import ceil, floor
from pathlib import Path


def chunk_textgrid_parser(textgrid_obj):
    """ Parsses the Textgrid of One Chunk and Returns a List

    Parameters
    ----------
    textgrid_obj : praatio.Textgrid()
        One Chunk's Textgrid Object imported using praatio

    Returns
    -------
    textgrid_dict : dict
        Dictionary of all of the Tiers for one Chunk's TextGrid Handlabels.
        If Chunk contains times that were previously labeled it will contain an additional key 'old_epochs'

        shape: {'labels' : [(start, end, label)],
                'female' : [(start, end, label)],
                'KWE' : [(timeV1, label1)],
                'old_epochs' : {'epoch_#' : [(start, end, label)]}
                }

    """
    textgrid_dict = {}  # Make empty dict

    # Make Sure the Required Tiers are present and in the correct order
    assert "labels" == textgrid_obj.tierNameList[0]
    assert "female" == textgrid_obj.tierNameList[1]
    assert "KWE" == textgrid_obj.tierNameList[2]

    # Make Dictionary of Required Tiers
    textgrid_dict["labels"] = _get_tier_list(textgrid_obj, "labels")  # Interval Tier of Chunk's Labels
    textgrid_dict["female"] = _get_tier_list(textgrid_obj, "female")  # Interval Tier of Female Sounds
    textgrid_dict["KWE"] = _get_tier_list(textgrid_obj, "KWE")  # Point Tier of Automated Labels

    # Add Optional Tier(s) to the Dictionary
    print(textgrid_obj.tierNameList)
    if len(textgrid_obj.tierNameList) > 3:
        old_epochs_dict = {}
        for index in range(3, len(textgrid_obj.tierNameList)):
            epoch_id = textgrid_obj.tierNameList[index]  # get the epoch tier name
            old_epochs_dict[epoch_id] = _get_tier_list(textgrid_obj, epoch_id)  # Interval Tier of old epoch's Labels
        textgrid_dict["old_epochs"] = old_epochs_dict  # Nested Dictionary of Interval Tiers

    return textgrid_dict


def _get_tier_list(textgrid_obj, tier_name: str):
    """ Get the List of Labels from the Textgrid Object"""

    tier = textgrid_obj.tierDict[tier_name]
    return tier.entryList


def _sanity_check_labels_tier(tier_list: list, chunk_num: int):
    """ Check the Labels in the labels tier"""

    for start, stop, label in tier_list:
        if label == 'BUFFER':
            pass  # Buffer useful when filtering Data so that phase distortion doesn't occur during times of interest
        elif label == '':
            raise ValueError("Empty Label in The Current Chunks's TextGrid: ", chunk_num, "Starts at: ", start)
        elif len(label) > 1:
            if label == 10:
                pass
            else:
                raise ValueError("Incorrect Label in the Current Chunks's TextGrid: ", chunk_num, "Starts at: ", start)
        else:
            pass


def conv_textgrid_to_dict(bird_id, session, base_folder: Path):
    """ Algorithm to Open all of the TextGrids, Convert them to Dictionaries and Append them to a List

    Returns
    -------
    days_handlabels_dict : dict
        Dictionary of all of the Tiers for all Chunks' TextGrid Handlabels.
        If Chunk contains times that were previously labeled it will contain a additional key 'old_epochs'

        shape : {Chunks Number : { 'labels' : [(start, end, label)],
                                  'female' : [(start, end, label)],
                                  'KWE' : [(timeV1, label1)],
                                  'old_epochs' : {'epoch_#' : [(start, end, label)]}
                                  }
    Notes
    -----
    From Praatio tutorial 1:
        For a pointTier, the entryList looks like:
            [(timeV1, label1), (timeV2, label2), ...]

        While for an intervalTier, the entryList looks like:
            [(startV1, endV1, label1), (startV2, endV2, label2), ...]
    """

    # 1. Navigate to Directory of Textgrids
    textgrid_dir = base_folder / bird_id / session

    # 2. Make a list of the Textgrids to be Converted
    textgrids = [i for i in textgrid_dir.iterdir() if i.suffix == ".TextGrid" and i.stem.isdigit()]  # Get path list
    textgrids = sorted(textgrids)  # Sort List in chronological order

    # 3. Iterate through each TextGrid and Convert to Dictionary
    days_handlabels_dict = {}  # Initiate empty dictionary

    for chunk in textgrids:
        chunk_id = int(chunk.stem)  # Get the Number/ID of Chunk
        print(chunk_id)
        chunk_textgrid = tgio.openTextgrid(str(chunk))  # Open Chunk's Textgrid
        chunk_dict = chunk_textgrid_parser(chunk_textgrid)  # Convert Chunk's Textgrid to dict
        _sanity_check_labels_tier(tier_list=chunk_dict['labels'], chunk_num=chunk_id)  # Sanity Check labels tier
        days_handlabels_dict[chunk_id] = chunk_dict  # Add Chunk's TextGrid->dict to Day's dictionary

    return days_handlabels_dict


def textgrid_dict_to_handlabels_list(days_handlabels_dict):
    """ Converts the Textgrid Dictionary of All chunks for one day into a list of BirdSongToolbox friendly dictionaries

    Parameters
    ----------
    days_handlabels_dict : dict
        Dictionary of all of the Tiers for all Chunks' TextGrid Handlabels.
        If Chunk contains times that were previously labeled it will contain a additional key 'old_epochs'
        shape : {Chunks Number : { 'labels' : [(start, end, label)],
                                  'female' : [(start, end, label)],
                                  'KWE' : [(timeV1, label1)],
                                  'old_epochs' : {'epoch_#' : [(start, end, label)]}
                                  }

    Returns
    -------
    handlabels : list
        List of all Dictionaries containing all of the handlabels for each chunk for one day
        shape: [chunk_num] -> {'labels' : (labels, onsets),
                               'female' : (labels, onsets),
                               'KWE' : [time],
                               'old_epochs' : {'epoch_#' : (labels, onsets)} # Only if Labeled the old way
                               }

    """

    handlabels = []

    for chunk_num in range(len(days_handlabels_dict.keys())):
        chunk_dict = chunk_textgrid_dict_to_handlabels_dict(days_handlabels_dict[chunk_num])
        handlabels.append(chunk_dict)

    return handlabels


def chunk_textgrid_dict_to_handlabels_dict(chunk_dict):
    """ Converts on Chunk's Textgrid Dictionary into a BirdSongToolbox friendly dictionary

    Parameters
    ----------
    chunk_dict : dict
        Dictionary of all of the Tiers for one Chunk's TextGrid Handlabels.
        If Chunk contains times that were previously labeled it will contain an additional key 'old_epochs'
        shape: {'labels' : [(start, end, label)],
                'female' : [(start, end, label)],
                'KWE' : [(timeV1, label1)],
                'old_epochs' : {'epoch_#' : [(start, end, label)]}
                }

    Returns
    -------
    handlabels_dict : dict
        Dictionary of all of the tiers of Handlabels for the given chunk
        shape: {'labels' : (labels, onsets),
                'female' : (labels, onsets),
                'KWE' : [time],
                'old_epochs' : {'epoch_#' : (labels, onsets)} # Only if Labeled the old way
                }
    """

    handlabels_dict = {}

    # Make Dictionary of Required Tiers
    handlabels_dict["labels"] = _conv_intervals_to_handlabels(chunk_dict["labels"])  # Chunk's Labels
    handlabels_dict["female"] = _conv_intervals_to_handlabels(chunk_dict["female"])  # Chunk's Female Sounds
    handlabels_dict["KWE"] = _conv_points_to_handlabels(chunk_dict["KWE"])  # Chunk's Automated Motif Onsets

    # Add Optional Old Epochs(KWE) labels to the Dictionary
    if 'old_epochs' in chunk_dict.keys():
        old_epochs_dict = {}
        for epoch in chunk_dict['old_epochs'].keys():
            old_epochs_dict[epoch] = _conv_intervals_to_handlabels(chunk_dict['old_epochs'][epoch])
        handlabels_dict['old_epochs'] = old_epochs_dict  # Optional Nested Dictionary of Old Epochs(KWE) labels

    return handlabels_dict


def _conv_intervals_to_handlabels(interval_list: list):
    """Converts the interval lists to a Format Useful to BirdSongToolbox

    Returns
    -------
    labels : list, shape: [Event_Labels]
        list of labels for one tier
    onsets : list, shape: [[Starts], [Ends]]
        list of start and end times for on tier
    """
    # Initiate lists
    labels = []
    starts = []
    ends = []

    # Iterate Through All Labels
    for start, stop, label in interval_list:
        starts.append(int(ceil(start * 30000)))  # ceil used to work around precision of Indexing
        ends.append(int(floor(stop * 30000)))  # floor used to work around precision of Indexing
        if label.isdigit():
            label = int(label)  # Change Integer Labels into int
        labels.append(label)

    onsets = [starts, ends]

    return labels, onsets


def _conv_points_to_handlabels(point_list: list):
    """Converts the point lists to a Format Useful to BirdSongToolbox

    Returns
    -------
    times : list, shape: [Times]
        Times of the Point Tier
    """

    times = []
    for time, _ in point_list:
        times.append(int(round(time * 30000)))  # Convert to Sample Rate and Round to the nearest Integer (Samples)

    times.append(time)
    return times

