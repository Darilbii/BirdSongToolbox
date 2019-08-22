""" Test the context_hand_labeling module"""

import pytest

from BirdSongToolbox.context_hand_labeling import ContextLabels
from BirdSongToolbox.file_utility_functions import _load_pckl_data
from BirdSongToolbox.free_epoch_tools import get_chunk_handlabels

@pytest.mark.run(order=1)
@pytest.fixture()
def bird_id():
    return 'z007'

@pytest.mark.run(order=1)
@pytest.fixture()
def session():
    return 'day-2016-09-09'



@pytest.mark.run(order=1)
def test_ContextLabels(bird_id, session, chunk_data_path):
    labels = _load_pckl_data(data_name="chunk_handlabels_Song", bird_id=bird_id, session=session,
                             source=chunk_data_path)

    chunk_labels, chunk_onsets = get_chunk_handlabels(labels)

    bout_states = {8: 'not', 'I': 'not', 'C': 'not', 1: 'bout', 2: 'bout', 3: 'bout', 4: 'bout', 5: 'bout', 6: 'bout',
                   7: 'bout', "BUFFER": "not", "X": "not"}
    bout_transitions = {'not': 1, 'bout': 8}
    bout_syll_length = 5
    ZebraContext = ContextLabels(bout_states, bout_transitions, full_bout_length=bout_syll_length)

    contexts = ZebraContext.get_all_context_index_arrays(chunk_labels)

