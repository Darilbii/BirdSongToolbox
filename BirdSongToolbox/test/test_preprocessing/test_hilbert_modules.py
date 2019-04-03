from BirdSongToolbox.ImportClass import Import_PrePd_Data
from BirdSongToolbox.PreProcessClass import BPF_Master, BPF_Module, hilbert_module
from BirdSongToolbox.config.settings import TEST_DATA_DIR

import numpy as np
import pytest


bird_id = 'z020'
date = 'day-2016-06-02'


@pytest.mark.run(order=1)
def test_hilbert_module_phase():
    PreP_Data = Import_PrePd_Data(bird_id, date, location=TEST_DATA_DIR)

    Channels = PreP_Data.Song_Neural
    Freq_Bands = ([10], [1])
    SN_L = PreP_Data.Sn_Len
    Gp_L = PreP_Data.Gap_Len
    Num_Chan = PreP_Data.Num_Chan
    order_num = 175
    fs = PreP_Data.Fs
    FiltFilt = True
    Num_Trials = PreP_Data.Num_Motifs

    song_length, number_channels = np.shape(Channels[0])
    tops, bottoms = Freq_Bands
    num_freq = len(tops)

    assert number_channels == Num_Chan

    Frequencies = BPF_Master(Channels, Num_Trials=Num_Trials, Freq_Bands=Freq_Bands, SN_L=SN_L, Gp_L=Gp_L,
                             Num_Chan=Num_Chan, Num_Freq=len(tops), order_num=order_num, fs=fs, FiltFilt=FiltFilt,
                             verbose=False)

    hilbert_results = hilbert_module(Frequencies, output='phase')

    # Smoke Tests: hilbert_results
    assert isinstance(hilbert_results, list)
    assert isinstance(hilbert_results[0], list)
    assert isinstance(hilbert_results[0][0], np.ndarray)
    assert np.shape(hilbert_results) == (Num_Trials, number_channels, song_length, num_freq)

    # # Unit Test: hilbert_results
    # assert hilbert_results[0][0, 0] == Frequencies[0][0, 0] - avg_freq_bins_lfp[0, 0]
    # assert hilbert_results[0][-1, 0] == Frequencies[0][-1, 0] - avg_freq_bins_lfp[-1, 0]
    # assert hilbert_results[-1][0, 0] == Frequencies[-1][0, 0] - avg_freq_bins_lfp[0, 0]
    # assert hilbert_results[-1][-1, 0] == Frequencies[-1][-1, 0] - avg_freq_bins_lfp[-1, 0]