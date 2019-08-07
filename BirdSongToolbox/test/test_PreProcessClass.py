"""Test functions for the ImportClass """

from BirdSongToolbox.ImportClass import Import_PrePd_Data
from BirdSongToolbox.PreProcessClass import Pipeline
from BirdSongToolbox.PreProcTools import Good_Channel_Index
import numpy as np
import pytest



bird_id = 'z020'
date = 'day-2016-06-02'


@pytest.mark.run(order=1)
def test_Pipeline_initiation(PrePd_data_dir_path):
    """ Test the bpf_module function"""
    #TODO: Make this more than a smoke test

    PreP_Data = Import_PrePd_Data(bird_id, date, location=PrePd_data_dir_path)

    pipeline = Pipeline(PreP_Data)

    # Smoke Tests
    assert isinstance(pipeline.Activity_Log, dict)
    assert isinstance(pipeline.Backup, tuple)
    assert isinstance(pipeline.Status, bool)

    # Unit Tests
    assert pipeline.bird_id == PreP_Data.bird_id
    assert pipeline.date == PreP_Data.date
    assert pipeline.Sn_Len == PreP_Data.Sn_Len
    assert pipeline.Gap_Len == PreP_Data.Gap_Len
    assert pipeline.Num_Chan == PreP_Data.Num_Chan
    assert pipeline.Bad_Channels == PreP_Data.Bad_Channels  # Debating Hard Passing Bad_Channels
    assert pipeline.Fs == PreP_Data.Fs
    assert np.all(pipeline.Song_Audio[0] == PreP_Data.Song_Audio[0])  # Debating Including Audio
    assert np.all(pipeline.Song_Neural[0] == PreP_Data.Song_Neural[0])
    assert np.all(pipeline.Silence_Audio[0] == PreP_Data.Silence_Audio[0])  # Debating Including Audio
    assert np.all(pipeline.Silence_Neural[0] == PreP_Data.Silence_Neural[0])
    assert pipeline.Num_Motifs == PreP_Data.Num_Motifs
    assert pipeline.Num_Silence == PreP_Data.Num_Silence
    assert np.all(pipeline.Good_Motifs == PreP_Data.Good_Motifs)
    assert np.all(pipeline.Bad_Motifs == PreP_Data.Bad_Motifs)
    assert np.all(pipeline.LS_Drop == PreP_Data.LS_Drop)
    assert np.all(pipeline.Last_Motifs == PreP_Data.Last_Motifs)
    assert np.all(pipeline.First_Motifs == PreP_Data.First_Motifs)
    assert np.all(pipeline.All_First_Motifs == PreP_Data.All_First_Motifs)
    assert np.all(pipeline.All_Last_Motifs == PreP_Data.All_Last_Motifs)
    assert np.all(pipeline.Good_Mid_Motifs == PreP_Data.Good_Mid_Motifs)
    assert np.all(pipeline.Good_Channels == Good_Channel_Index(PreP_Data.Num_Chan, PreP_Data.Bad_Channels))
    assert pipeline.Status
    assert pipeline.Step_Count == 0

#TODO: Write test funtions for all of the internal mechanisms of the Pipeline Class

