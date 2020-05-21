"""
Get the First Syllable of Bout (Bout Duration greater than 1)
=============================================================

This example shows how to use the contextual labels to get specific event times
"""

###############################################################################
from BirdSongToolbox.import_data import ImportData
import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.context_hand_labeling import ContextLabels, label_focus_context
from BirdSongToolbox.behave.behave_utils import event_array_maker_chunk, get_events_rasters, repeat_events

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import inspect

###############################################################################


# Select bird_id and session
bird_id = 'z007'
session = 'day-2016-09-09'

###############################################################################

src_file_path = inspect.getfile(lambda: None)
project_dir = Path(src_file_path).resolve().parents[1]
print(project_dir)
data_dir = project_dir / "BirdSongToolbox" / "data" / "Chunk_Data_Demo"

# Import Data
zdata = ImportData(bird_id=bird_id, session=session, location=data_dir)

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

###############################################################################

first_event_times = fet.make_event_times_axis(first_window, fs=1000)

###############################################################################


# Create timeseries representing the labeled Events For all Chunks
event_array_test2 = event_array_maker_chunk(labels_list=chunk_labels_list, onsets_list=chunk_onsets_list)

###############################################################################

first_events = get_events_rasters(data=event_array_test2, indices=first_syll, fs=1000, window=first_window)
fill_events_first = repeat_events(first_events)

###############################################################################


def plot_behavior_test(fill_events_context, context_event_times, context_events, ax=None):
    # Setup the Colorbar

    cmap2 = matplotlib.colors.ListedColormap(
        ['#000000', '#B66DFF', '#db6e00', '#009292', '#924900', '#006DDB', '#B6DBFF', 'white', '#feb4d9', '#490092'])
    cmap2.set_over('cyan')
    cmap2.set_under('#B6DBFF')

    bounds = [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap2.N)

    # PlotBehavior Raster
    num_events = context_events.shape[0]
    max_len = fill_events_context.shape[0]
    bin_width = (max_len) / (num_events)
    y_labels = np.arange(0, num_events, 5, dtype=int)
    y_steps = np.linspace(0, y_labels[-1] * bin_width, len(y_labels), dtype=int)
    y_steps[1:] = y_steps[1:] - int(bin_width / 2)

    if ax is None:
        plt.imshow(fill_events_context, cmap=cmap2, Norm=norm, aspect="auto")
        plt.yticks(ticks=y_steps[1:], labels=y_labels[1:])
        plt.ylim(0, max_len)

    else:
        ax.imshow(fill_events_context, cmap=cmap2, Norm=norm, aspect="auto")
        ax.set_yticks(y_steps[1:])
        ax.set_yticklabels(y_labels[1:])
        ax.set_ylim(0, max_len)
        ax.set_xticks([])


###############################################################################

# Plot First Behavior Raster
plot_behavior_test(fill_events_context=fill_events_first, context_event_times=first_event_times,
                   context_events=first_events)

# ax[0].set_title(label="Start of Bout", fontsize=bigsize)
# ax[0].set_ylabel(ylabel='Bout #', fontsize=subsize)
# ax[0].tick_params(axis='both', which='major', labelsize=ticksize)

###############################################################################