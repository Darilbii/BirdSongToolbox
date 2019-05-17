""" Functions for assigining additional context dependent labels to hand labels of auido recording (Vocal Behavior)"""
import numpy as np


class ContextLabels(object):
    #     bout_breaks = {'not': 'start', 'bout': 'end'}

    def __init__(self, bout_states, bout_transitions, full_bout_length: int):
        """

        :param bout_states:
        :param bout_transitions:
        :param full_bout_length:

        # Example from z020:
        >>>bout_states = {8:'not', 'I':'not','C':'not', 1:'bout',2:'bout',3:'bout',4:'bout',5:'bout',6:'bout',7:'bout'}
        >>>bout_transitions = {'not':1,'bout':8}
        >>>bout_syll_length = 4
        >>>testclass = ContextLabels(bout_states, bout_transitions, full_bout_length = 4)

        """
        self.bout_states = bout_states
        self.bout_transitions = bout_transitions
        self.bout_end_syllable = full_bout_length

    def _get_bout_state(self, current_label):
        return self.bout_states[current_label]

    #     def bout_level(self, labels: list):
    #         "Returns list of the Start and End of the Bout"
    #         motif_results = []
    #         # Make a padded list assuming no transitions at edges
    #         labels_padded = [labels[0]] + labels[:-1]

    #         # For Each Label
    #         for index, current_label in enumerate(labels):
    #             previous_state = self._get_bout_state(labels_padded[index])  # Transition depends on previous state

    #             # If the current label is the transition for the previous state
    #             if current_label == self.bout_transitions[previous_state]:
    #                 motif_results.append(self.bout_breaks[previous_state])
    #             else:
    #                 motif_results.append(0)

    #         return motif_results

    def bout_array(self, labels: list):
        """Returns and array that has 1 for every bout and a 0 for everything else for one epoch"""

        bout_results = []

        # For Each Label
        for index, current_label in enumerate(labels):
            current_state = self._get_bout_state(current_label)  # Transition depends on previous state

            # If the current state is bout
            if current_state == 'bout':
                bout_results.append(1)  # Motif
            else:
                bout_results.append(0)  # Everything else

        return bout_results

    def bout_index(self, labels: list):
        """ Gets the index for the labels that are the start and end of each Bout for one epoch
        """

        bout_results = self.bout_array(labels=labels)

        ones = [index for index, value in enumerate(bout_results) if value == 1]

        # Get Starts
        prior_padded = [bout_results[0]] + bout_results  # Pad with the first label
        starts = [index for index in ones if prior_padded[index] == 0]  # Get the front edge

        # Get Ends
        post_padded = bout_results[1:] + [bout_results[-1]]  # Pad with the last label
        ends = [index for index in ones if post_padded[index] == 0]  # Get the end edge

        print('starts:', starts)
        print('ends:', ends)

        # Check for Worst Case Scenario: The Center Bout extends Beyound the Epoch
        if len(ends) < 1:
            starts = []
        else:
            # Cut Out Bout Indexes that can't be resolved within the labeled Epoch (Edge Cases)
            if starts[0] > ends[0]:
                ends = np.delete(ends, 0)  # Delete unresolvable end of Bout that continues beyond the labeled Epoch
            if starts[-1] > ends[-1]:
                starts = np.delete(starts,
                                   0)  # Delete unresolvable start of Bout that continues beyond the labeled Epoch

        return starts, ends

    def motif_array(self, labels: list):
        """Returns an array that has 1 for every motif and 2 for every intra-motif silence for one epoch"""
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
        """ Gets the index for the labels that are the start and end of each Motif for one epoch
        """

        motif_results = self.motif_array(labels=labels)

        ones = [index for index, value in enumerate(motif_results) if value == 1]

        # Get Starts
        prior_padded = [motif_results[0]] + motif_results  # Pad with the first label
        starts = [index for index in ones if prior_padded[index] == 0 or prior_padded[index] == 2]

        # Get Ends
        post_padded = motif_results[1:] + [motif_results[-1]]  # Pad with the last label
        ends = [index for index in ones if post_padded[index] == 0 or post_padded[index] == 2]

        return starts, ends

    def _motif_sequence_in_bout_array(self, labels: list, bout_starts: list, bout_ends: list, motif_starts: list,
                                      motif_ends: list):
        # Makes an array that indexes the Motif Seqeunce Number within Bout (Based on the Label Identity)

        motif_array = np.zeros((len(labels)))

        for bout_start, bout_end in zip(bout_starts, bout_ends):
            motif_number = 1
            for motif_start, motif_end in zip(motif_starts, motif_ends):
                if motif_start >= bout_start and motif_end <= bout_end:
                    motif_array[motif_start:motif_end + 1] = int(motif_number)
                    motif_number += 1

        return motif_array

    def _first_motif_in_bout_array(self, labels: list, bout_starts: list, motif_starts: list, motif_ends: list):
        # Makes an array that indexes the labels that occur during All First Motifs that can be resolved in one Epoch

        motif_array = np.zeros((len(labels)))

        for bout_start in bout_starts:
            for motif_start, motif_end in zip(motif_starts, motif_ends):
                if motif_start == bout_start:
                    motif_array[motif_start:motif_end + 1] = 1

        return motif_array

    def _last_motif_in_bout_array(self, labels: list, bout_ends: list, motif_starts: list, motif_ends: list):
        # Makes an array that indexes the labels that occur during All Last Motifs that can be resolved in one Epoch

        motif_array = np.zeros((len(labels)))

        for bout_end in bout_ends:
            for motif_start, motif_end in zip(motif_starts, motif_ends):
                if motif_end == bout_end:
                    motif_array[motif_start:motif_end + 1] = 1

        return motif_array

    def _last_syllable_dropped_in_bout_array(self, labels: list, bout_starts: list, bout_ends: list, motif_starts: list,
                                             motif_ends: list):
        # Makes an array that indexes the Motif with syllables skipped within Bout (Based on the Label Identity)

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
        # Note this can only get information that can be resolved within the Epoch. Bout Edge-cases are ignored

        motif_array = np.zeros((len(labels), 4))

        bout_starts, bout_ends = self.bout_index(labels=labels)
        motif_starts, motif_ends = self._motif_index(labels=labels)

        # Get Motif Sequence Number withing Bout
        motif_array[:, 0] = self._motif_sequence_in_bout_array(labels=labels, bout_starts=bout_starts,
                                                               bout_ends=bout_ends, motif_starts=motif_starts,
                                                               motif_ends=motif_ends)

        # Get All First Motifs in Epoch
        motif_array[:, 1] = self._first_motif_in_bout_array(labels=labels, bout_starts=bout_starts,
                                                            motif_starts=motif_starts, motif_ends=motif_ends)

        # Get All Last Motifs in Epoch
        motif_array[:, 2] = self._last_motif_in_bout_array(labels=labels, bout_ends=bout_ends,
                                                           motif_starts=motif_starts, motif_ends=motif_ends)

        # Get All Motifs in Bouts with Skipped (Last) Syllables
        motif_array[:, 3] = self._last_syllable_dropped_in_bout_array(labels=labels, bout_starts=bout_starts,
                                                                      bout_ends=bout_ends, motif_starts=motif_starts,
                                                                      motif_ends=motif_ends)

        return motif_array

    def get_all_context_index_arrays(self, all_labels: list):
        # get Context Arrays for all the Epoch for a day

        motif_array_list = []

        for labels in all_labels:
            motif_array_list.append(self.get_context_index_array(labels=labels))

        return motif_array_list
