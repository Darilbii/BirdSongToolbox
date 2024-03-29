""" Functions for assigining additional context dependent labels to hand labels of auido recording (Vocal Behavior)"""
import numpy as np


class ContextLabels(object):
    """ Class which adds additional labels on top of the handlables to allow for flexible selection of events using
    context
    """
    #     bout_breaks = {'not': 'start', 'bout': 'end'}

    def __init__(self, bout_states: dict, bout_transitions: dict, full_bout_length: int):
        """ Create Contextual Labels for the Handlabels based on the defined behavior of the birds song

        Parameters
        ----------
        bout_states : dict
            dictionary of all labels used for specified bird and their context for vocal behavior
        bout_transitions : dict
            dictionary of the transition label for each state (bout & not)
        full_bout_length : int
            Last syllable of the stereotyped portion of the Motif (Not the Intra-Motif note)

        Examples
        --------

        Using behavior from subject z020:

        >>> bout_states = {8:'not', 'I':'not','C':'not', 1:'bout',2:'bout',3:'bout',4:'bout',5:'bout',6:'bout',7:'bout'}
        >>> bout_transitions = {'not':1,'bout':8}
        >>> bout_syll_length = 4
        >>> testclass = ContextLabels(bout_states, bout_transitions, bout_syll_length)

        """
        self.bout_states = bout_states
        self.bout_transitions = bout_transitions
        self.bout_end_syllable = full_bout_length

    def _get_bout_state(self, current_label):
        return self.bout_states[current_label]

    def bout_array(self, labels: list):
        """Returns an array that has 1 for every bout and a 0 for everything else for one chunk

        Parameters
        ----------
        labels : list | [Labels]
            list of all labels for one Epoch

        Returns
        -------
        bout_results : list
            Array that encodes each label as either bout or not bout using 1 or 0, respectively.
        """

        bout_results = []

        # For Each Label
        for index, current_label in enumerate(labels):
            current_state = self._get_bout_state(current_label)  # Transition depends on previous state

            # If the current state is bout
            if current_state == 'bout':
                bout_results.append(1)  # Song
            else:
                bout_results.append(0)  # Everything else

        return bout_results

    def bout_index(self, labels: list):
        """ Gets the index for the labels that are the start and end of each Bout for one epoch

        Parameters
        ----------
        labels : list | [Labels]
            list of all labels for one Epoch

        Returns
        -------
        starts : list
            List of all Start Times corresponding to each motif in one chunk
        ends : list
            List of all End Times corresponding to each motif in one chunk
        """

        bout_results = self.bout_array(labels=labels)

        ones = [index for index, value in enumerate(bout_results) if value == 1]

        # Get Starts
        prior_padded = [bout_results[0]] + bout_results  # Pad with the first label
        starts = [index for index in ones if prior_padded[index] == 0]  # Get the front edge

        # Get Ends
        post_padded = bout_results[1:] + [bout_results[-1]]  # Pad with the last label
        ends = [index for index in ones if post_padded[index] == 0]  # Get the end edge

        return starts, ends

    def motif_array(self, labels: list):
        """Returns an array that has 1 for every motif and 2 for every intra-motif silence for one chunk

        Parameters
        ----------
        labels : list | [Labels]
            list of all labels for one Epoch

        Returns
        -------
        motif_results : ndarray
            array that segments motifs using both the labels and the initialized parameters for the bird's song
            structure
        """
        motif_results = []

        labels_padded = labels + [labels[-1]]  # Make a padded list assuming no transitions at edges

        # For Each Label
        for index, current_label in enumerate(labels):
            current_state = self._get_bout_state(current_label)  # Transition depends on previous state

            # If the current state is bout
            if current_state == 'bout':
                if labels_padded[index + 1] == 1:
                    motif_results.append(2)  # Intra-Motif Silence / End of Motif
                else:
                    motif_results.append(1)  # Motif
            else:
                motif_results.append(0)  # Everything else

        return motif_results

    def _motif_index(self, labels: list):
        """ Gets the index for the labels that are the start and end of each Motif for one chunk
        """

        motif_results = self.motif_array(labels=labels)

        ones = [index for index, value in enumerate(motif_results) if value == 1]

        # Get Starts
        prior_padded = [motif_results[0]] + motif_results  # Pad with the first label
        starts = [index for index in ones if prior_padded[index] == 0 or prior_padded[index] == 2]

        # Get Ends
        post_padded = motif_results[1:] + [motif_results[-1]]  # Pad with the last label
        ends = [index for index in ones if post_padded[index] == 0 or post_padded[index] == 2]

        # Cut Out Motif Indexes that can't be resolved within the labeled Epoch (Edge Cases)
        if starts[0] > ends[0]:
            ends = np.delete(ends, 0)  # Delete unresolvable end of Motif that continues beyond the labeled Epoch
        if starts[-1] > ends[-1]:
            starts = np.delete(starts, -1)  # Delete unresolvable start of Motif that continues beyond the labeled Epoch

        return starts, ends

    def _edge_corrections(self, starts: list, ends: list):
        """Eliminate labels of bouts that can't be resolved within the Epoch (kwe)
        """

        # Check for Worst Case Scenario: The Center Bout extends Beyound the Epoch
        if len(ends) < 1:
            starts = []
        else:
            # Cut Out Bout Indexes that can't be resolved within the labeled Epoch (Edge Cases)
            if starts[0] > ends[0]:
                ends = np.delete(ends, 0)  # Delete end of Bout that continues beyond the labeled Epoch
            if starts[-1] > ends[-1]:
                starts = np.delete(starts, -1)  # Delete start of Bout that continues beyond the labeled Epoch
        return starts, ends

    def _motif_sequence_in_bout_array(self, labels: list, bout_starts: list, bout_ends: list, motif_starts: list,
                                      motif_ends: list):
        """Makes an array that indexes the Motif Seqeunce Number within Bout (Based on the Label Identity)
        """

        motif_array = np.zeros((len(labels)))

        for bout_start, bout_end in zip(bout_starts, bout_ends):
            motif_number = 1
            for motif_start, motif_end in zip(motif_starts, motif_ends):
                if motif_start >= bout_start and motif_end <= bout_end:
                    motif_array[motif_start:motif_end + 1] = int(motif_number)
                    motif_number += 1

        return motif_array

    def _first_motif_in_bout_array(self, labels: list, bout_starts: list, motif_starts: list, motif_ends: list):
        """Makes an array that indexes the labels that occur during All First Motifs that are resolved in one Chunk
        """

        motif_array = np.zeros((len(labels)))

        for bout_start in bout_starts:
            for motif_start, motif_end in zip(motif_starts, motif_ends):
                if motif_start == bout_start:
                    motif_array[motif_start:motif_end + 1] = 1

        return motif_array

    def _last_motif_in_bout_array(self, labels: list, bout_ends: list, motif_starts: list, motif_ends: list):
        """Makes an array that indexes the labels that occur during All Last Motifs that can be resolved in one Epoch"""

        motif_array = np.zeros((len(labels)))

        for bout_end in bout_ends:
            for motif_start, motif_end in zip(motif_starts, motif_ends):
                if motif_end == bout_end:
                    motif_array[motif_start:motif_end + 1] = 1  # Mark the Last Motif in the Bout

        return motif_array

    def _last_syllable_dropped_in_bout_array(self, labels: list, bout_starts: list, bout_ends: list, motif_starts: list,
                                             motif_ends: list):
        """Makes an array that indexes the Motif with syllables skipped within Bout (Based on the Label Identity)
        """

        # TODO: If there is more variation in which syllable is dropped then use list comprehension to see if any ...
        # syllables labels of a full list are not in the labels between the start and end of the motif ...
        # example [1,2,3,4] for z020

        motif_array = np.zeros((len(labels)))

        for bout_start, bout_end in zip(bout_starts, bout_ends):
            for motif_start, motif_end in zip(motif_starts, motif_ends):
                if motif_start >= bout_start and motif_end <= bout_end:
                    if labels[motif_end] != self.bout_end_syllable:
                        motif_array[motif_start:motif_end + 1] = 1

        return motif_array

    def get_context_index_array(self, labels: list):
        """ Get Context array for one Epoch

        Note
        ----
        This can only get information that can be resolved within the Epoch. Bout Edge-cases are ignored

        Parameters
        ----------
        labels : list | [Labels]
            list of all labels for one Epoch

        Returns
        -------
        motif_array : array
            Array of context labels for one Epoch.
            columns: (Motif Sequence in Bout, First Motif (1-hot), Last Motif (1-hot), Last Syllable Dropped (1-hot))
        """

        motif_array = np.zeros((len(labels), 4))

        bout_starts, bout_ends = self.bout_index(labels=labels)
        fix_bout_starts, fix_bout_ends = self._edge_corrections(starts=bout_starts, ends=bout_ends)
        motif_starts, motif_ends = self._motif_index(labels=labels)

        # Get Motif Sequence Number withing Bout
        motif_array[:, 0] = self._motif_sequence_in_bout_array(labels=labels, bout_starts=fix_bout_starts,
                                                               bout_ends=fix_bout_ends, motif_starts=motif_starts,
                                                               motif_ends=motif_ends)

        # Get All First Motifs in Epoch
        motif_array[:, 1] = self._first_motif_in_bout_array(labels=labels, bout_starts=bout_starts,
                                                            motif_starts=motif_starts, motif_ends=motif_ends)

        # Get All Last Motifs in Epoch
        motif_array[:, 2] = self._last_motif_in_bout_array(labels=labels, bout_ends=bout_ends,
                                                           motif_starts=motif_starts, motif_ends=motif_ends)

        # Get All Motifs in Bouts with Skipped (Last) Syllables
        motif_array[:, 3] = self._last_syllable_dropped_in_bout_array(labels=labels, bout_starts=fix_bout_starts,
                                                                      bout_ends=fix_bout_ends,
                                                                      motif_starts=motif_starts, motif_ends=motif_ends)

        return motif_array

    def get_all_context_index_arrays(self, all_labels: list):
        """Get Context Arrays for all the Epoch for a day

        Note
        ----
        this can only get information that can be resolved within the Epoch. Bout Edge-cases are ignored

        Parameters
        ----------
        all_labels : list | [Labels]
            list of all labels for one Epoch

        Returns
        -------
        motif_array_list : list
            list of arrays of context labels for each Epoch.
            [Epoch #] -> (labels, 4)
                col: (Motif Sequence in Bout, First Motif (1-hot), Last Motif (1-hot), Last Syllable Dropped (1-hot))

        """

        motif_array_list = []

        for labels in all_labels:
            motif_array_list.append(self.get_context_index_array(labels=labels))

        return motif_array_list


# Copied from Inter-Tiral Coherence Notebook
# TODO: Write Tests for label_focus_context and the Sub Functions

def label_focus_context(focus, labels, starts, contexts, context_func):
    """ Create a list of every instance of the User defined User Label (Focus on One Label)

    Parameters
    ----------
    focus : str or int
        User defined Label to focus on
    labels : list
        List of all Labels corresponding to each Epoch in Full_Trials
        [Epochs]->[Labels]
    starts : list
        List of all Start Times corresponding to each Epoch in Full_Trials (Note: sampled at 30KHz)
        [Epochs]->[Start Time]
    contexts : list
        list of arrays of context labels for each Epoch.
        [Epoch #] -> (labels, 4)
            col: (Motif Sequence in Bout, First Motif (1-hot), Last Motif (1-hot), Last Syllable Dropped (1-hot))
    context_func : func
        function that returns a bool based on some criterion from the context labels

    Returns
    -------
    Label_Index : list
        List of all start frames of every instances of the label of focus
        [Num_Trials]->[Num_Exs]
    """
    Label_Index = []

    # for i in range(len(Labels)):
    #     Trial_Labels = [int(Starts[i][x] / 30) for x in range(len(Labels[i])) if Labels[i][x] == Focus]

    for start, epoch, context in zip(starts, labels, contexts):
        trial_labels = [start[i] for i, (x, order, first, last, ls_drop) in
                        enumerate(zip(epoch, context[:, 0], context[:, 1], context[:, 2], context[:, 3])) if
                        x == focus and context_func(order, first, last, ls_drop)]
        Label_Index.append(trial_labels)
    return Label_Index


# Define a Function that evaulates labels based on the Context Specified
def first_context_func(order, first, last, ls_drop):
    return first == 1 and last == 0


# Define a Function that evaulates labels based on the Context Specified
def last_context_func(order, first, last, ls_drop):
    return first == 0 and last == 1


# Define a Function that evaulates labels based on the Context Specified
def mid_context_func(order, first, last, ls_drop):
    return first == 0 and last == 0


def get_motif_identifier(focus, context, labels):
    """

    :param focus: list
    :param context:
    :param labels:
    :return:
    """

    focus_index = dict()

    for i in focus:
        focus_index[i] = []

    sequential_counter = -1


    for chunk_contexts, chunk_labels in zip(context, labels):
        current_counter = 0  # Make sure to reset the Current Counter with each Chunk

        for motif_seq_numb, curr_label in zip(chunk_contexts[:, 0], chunk_labels):

            if motif_seq_numb != 0:  # For the Duration of this Motif

                if motif_seq_numb != current_counter:  # If in a New Motif
                    current_counter = motif_seq_numb  # Update the Current Motif Recognition
                    sequential_counter += 1  # Increase Motif Counter
                    print(motif_seq_numb)
                    print(sequential_counter)

                if curr_label in focus:
                    focus_index[curr_label].append(sequential_counter)  # If the syll occurs in this Motif the index it
    return