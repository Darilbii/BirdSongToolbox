"""
Get the First Syllable of Bout (Bout Duration greater than 1)
=============================================================

This example shows how to use the contextual labels to get specific event times
"""


###############################################################################

from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.context_hand_labeling import ContextLabels, label_focus_context
import BirdSongToolbox.free_epoch_tools as fet



###############################################################################


# Select bird_id and session
bird_id = 'z007'
session = 'day-2016-09-09'


###############################################################################

# Import Data
zdata = ImportData(bird_id=bird_id, session=session)


###############################################################################

# Reshape Handlabels into Useful Format
chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)


###############################################################################

# Initialize and configure instance of ContextLabels object
bout_states = {'BUFFER': 'not', 'X': 'not', 8: 'not', 'I': 'not', 'C': 'not', 1: 'bout', 2: 'bout', 3: 'bout',
               4: 'bout', 5: 'bout', 6: 'bout', 7: 'bout', 9: 'bout'}

bout_transitions = {'not': 1, 'bout': 8}

full_bout_length = 5

testclass = ContextLabels(bout_states, bout_transitions, full_bout_length)


###############################################################################

# Get the Context Array for the Day's Data

test_context = testclass.get_all_context_index_arrays(chunk_labels_list)


###############################################################################
# Define a context based on the boolean structure of the contextual labels

# Define a Function that evaulates labels based on the Context Specified
def first_context_example(order, first, last, ls_drop):
    return first == 1 and last == 0


###############################################################################

# Select Labels Using Flexible Context Selection
first_syll = label_focus_context(focus=1, labels=chunk_labels_list, starts=chunk_onsets_list[0], contexts=test_context,
                                 context_func=first_context_example)

###############################################################################

# Set the Context Windows

first_window = (-500, 800)

###############################################################################


# # Clip around Events of Interest
# all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=first_syll,
#                                         fs=1000, window=first_window )

###############################################################################

# all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)
#
# print(np.shape(all_firsts))

