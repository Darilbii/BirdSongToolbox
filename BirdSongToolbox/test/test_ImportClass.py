"""Test functions for the ImportClass """

from BirdSongToolbox.config.settings import TEST_DATA_DIR
from BirdSongToolbox.ImportClass import Import_PrePd_Data
import numpy as np
import pytest


@pytest.mark.run(order=1)
def test_ImportClass():
    """ Test ImportClass"""

    bird_id = 'z020'
    date = 'day-2016-06-02'

    PreP_Data = Import_PrePd_Data(bird_id, date, location=TEST_DATA_DIR)

    # Making sure the class initiated properly
    assert isinstance(PreP_Data, Import_PrePd_Data)

    assert PreP_Data.bird_id == bird_id
    assert PreP_Data.date == date
    assert isinstance(PreP_Data.data_type, str)
    assert isinstance(PreP_Data.Sn_Len, int)
    assert isinstance(PreP_Data.Gap_Len, int)
    assert isinstance(PreP_Data.Fs, int)
    assert isinstance(PreP_Data.Num_Chan, int)
    assert isinstance(PreP_Data.Bad_Channels, list)
    assert isinstance(PreP_Data.Num_Motifs, int)
    assert isinstance(PreP_Data.Num_Silence, int)

    # Test Song Data
    assert isinstance(PreP_Data.Song_Neural, list)
    assert isinstance(PreP_Data.Song_Audio, list)
    assert len(PreP_Data.Song_Neural) == len(PreP_Data.Song_Audio) == PreP_Data.Num_Motifs #, f"Song_Neural len: {PreP_Data.Song_Neural} \n Song_Audio len: {PreP_Data.Song_Audio} \n Num_Motifs: {PreP_Data.Num_Motifs}"
    assert np.shape(PreP_Data.Song_Neural[0]) == np.shape(PreP_Data.Song_Neural[-1]) == (PreP_Data.Gap_Len+PreP_Data.Sn_Len, PreP_Data.Num_Chan)
    assert len(PreP_Data.Song_Audio[0]) == len(PreP_Data.Song_Audio[-1]) == len(PreP_Data.Silence_Audio[0]) == len(PreP_Data.Silence_Audio[-1])

    # Test Silence Data
    assert isinstance(PreP_Data.Silence_Neural, list)
    assert isinstance(PreP_Data.Silence_Audio, list)
    assert len(PreP_Data.Silence_Neural) == len(PreP_Data.Silence_Audio) == PreP_Data.Num_Silence #, f"Song_Neural len: {PreP_Data.Silence_Neural} \n Song_Audio len: {PreP_Data.Silence_Neural} \n Num_Motifs: {PreP_Data.Num_Silence}"
    assert np.shape(PreP_Data.Silence_Neural[0]) == np.shape(PreP_Data.Silence_Neural[-1]) == (PreP_Data.Gap_Len+PreP_Data.Sn_Len, PreP_Data.Num_Chan)


    # Test Labels (Pass 1)
    assert isinstance(PreP_Data.Song_Quality, list)
    assert len(PreP_Data.Song_Quality) == PreP_Data.Num_Motifs
    assert isinstance(PreP_Data.Song_Locations, list)
    assert len(PreP_Data.Song_Locations) == PreP_Data.Num_Motifs
    assert isinstance(PreP_Data.Song_Syl_Drop, list)
    assert len(PreP_Data.Song_Syl_Drop) == PreP_Data.Num_Motifs

    # Test Index (Pass 1)
    assert isinstance(PreP_Data.Good_Motifs, np.ndarray)
    assert isinstance(PreP_Data.First_Motifs, np.ndarray)
    assert isinstance(PreP_Data.Last_Motifs, np.ndarray)
    assert isinstance(PreP_Data.Bad_Motifs, np.ndarray)
    assert isinstance(PreP_Data.LS_Drop, np.ndarray)
    assert isinstance(PreP_Data.All_First_Motifs, np.ndarray)
    assert isinstance(PreP_Data.All_Last_Motifs, np.ndarray)
    assert isinstance(PreP_Data.Good_Mid_Motifs, np.ndarray)



