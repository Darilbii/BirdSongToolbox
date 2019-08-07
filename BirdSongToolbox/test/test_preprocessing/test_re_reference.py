from BirdSongToolbox.ImportClass import Import_PrePd_Data
from BirdSongToolbox.PreProcessClass import BPF_Master, BPF_Module, RR_Neural_Module, RR_Neural_Master
from BirdSongToolbox.PreProcTools import Good_Channel_Index
import numpy as np
import pytest


bird_id = 'z020'
date = 'day-2016-06-02'

@pytest.mark.run(order=1)
def test_rr_neural_module(PrePd_data_dir_path):
    PreP_Data = Import_PrePd_Data(bird_id, date, location=PrePd_data_dir_path)

    Channels = PreP_Data.Song_Neural[0]
    Freq_Bands = ([10], [1])
    SN_L = PreP_Data.Sn_Len
    Gp_L = PreP_Data.Gap_Len
    Num_Chan = PreP_Data.Num_Chan
    order_num = 175
    fs = PreP_Data.Fs
    FiltFilt = True
    Good_Channels = Good_Channel_Index(PreP_Data.Num_Chan, PreP_Data.Bad_Channels)

    song_length, number_channels = np.shape(Channels)
    tops, bottoms = Freq_Bands
    num_freq = len(tops)

    Frequencies = BPF_Module(Channels, Freq_Bands=Freq_Bands, SN_L=SN_L, Gp_L=Gp_L, Num_Chan=Num_Chan,
                             Num_Freq=num_freq, order_num=order_num, fs=fs, FiltFilt=FiltFilt)

    freq_bins_rr, avg_freq_bins_lfp = RR_Neural_Module(Frequencies, Good_Channels=Good_Channels, SN_L=SN_L, Gp_L=Gp_L,
                                                       Num_Freq=num_freq)

    # Smoke Tests: freq_bins_rr
    assert isinstance(freq_bins_rr, list)
    assert isinstance(freq_bins_rr[0], np.ndarray)
    assert np.shape(freq_bins_rr) == (number_channels, song_length, num_freq)

    # Smoke Test: avg_freq_nins_lfp
    assert isinstance(avg_freq_bins_lfp, np.ndarray)
    assert np.shape(avg_freq_bins_lfp) == (song_length, num_freq)

    # Unit Test: freq_bins_rr
    assert freq_bins_rr[0][0,0] == Frequencies[0][0,0] - avg_freq_bins_lfp[0,0]
    assert freq_bins_rr[0][-1,0] == Frequencies[0][-1,0] - avg_freq_bins_lfp[-1,0]
    assert freq_bins_rr[-1][0,0] == Frequencies[-1][0,0] - avg_freq_bins_lfp[0,0]
    assert freq_bins_rr[-1][-1,0] == Frequencies[-1][-1,0] - avg_freq_bins_lfp[-1,0]


@pytest.mark.run(order=1)
def test_rr_neural_master(PrePd_data_dir_path):
    PreP_Data = Import_PrePd_Data(bird_id, date, location=PrePd_data_dir_path)

    Channels = PreP_Data.Song_Neural
    Freq_Bands = ([10], [1])
    SN_L = PreP_Data.Sn_Len
    Gp_L = PreP_Data.Gap_Len
    Num_Chan = PreP_Data.Num_Chan
    Num_Trials = PreP_Data.Num_Motifs
    order_num = 175
    fs = PreP_Data.Fs
    FiltFilt = True

    Good_Channels = Good_Channel_Index(PreP_Data.Num_Chan, PreP_Data.Bad_Channels)
    song_length, number_channels = np.shape(Channels[0])
    tops, bottoms = Freq_Bands
    num_freq = len(tops)

    assert number_channels == Num_Chan

    Frequencies = BPF_Master(Channels, Num_Trials=Num_Trials, Freq_Bands=Freq_Bands, SN_L=SN_L, Gp_L=Gp_L,
                             Num_Chan=Num_Chan, Num_Freq=len(tops), order_num=order_num, fs=fs, FiltFilt=FiltFilt,
                             verbose=False)

    freq_bins_rr, avg_freq_bins_lfp = RR_Neural_Master(Frequencies, Num_Trials=Num_Trials, Good_Channels=Good_Channels,
                                                       SN_L=SN_L, Gp_L=Gp_L, Num_Freq=num_freq)

    # Smoke Tests: freq_bins_rr
    assert isinstance(freq_bins_rr, list)
    assert isinstance(freq_bins_rr[0], list)
    assert isinstance(freq_bins_rr[0][0], np.ndarray)
    assert np.shape(freq_bins_rr) == (Num_Trials, number_channels, song_length, num_freq)

    # Smoke Test: avg_freq_nins_lfp
    assert isinstance(avg_freq_bins_lfp, list)
    assert isinstance(avg_freq_bins_lfp[0], np.ndarray)
    assert np.shape(avg_freq_bins_lfp) == (Num_Trials, song_length, num_freq)

    # Unit Test: freq_bins_rr
    assert freq_bins_rr[0][0][0, 0] == Frequencies[0][0][0,0 ] - avg_freq_bins_lfp[0][0, 0]
    assert freq_bins_rr[0][0][-1, 0] == Frequencies[0][0][-1, 0] - avg_freq_bins_lfp[0][-1, 0]
    assert freq_bins_rr[0][-1][0, 0] == Frequencies[0][-1][0, 0] - avg_freq_bins_lfp[0][0, 0]
    assert freq_bins_rr[0][-1][-1, 0] == Frequencies[0][-1][-1, 0] - avg_freq_bins_lfp[0][-1, 0]
    assert freq_bins_rr[-1][0][0, 0] == Frequencies[-1][0][0,0 ] - avg_freq_bins_lfp[-1][0, 0]
    assert freq_bins_rr[-1][0][-1, 0] == Frequencies[-1][0][-1, 0] - avg_freq_bins_lfp[-1][-1, 0]
    assert freq_bins_rr[-1][-1][0, 0] == Frequencies[-1][-1][0, 0] - avg_freq_bins_lfp[-1][0, 0]
    assert freq_bins_rr[-1][-1][-1, 0] == Frequencies[-1][-1][-1, 0] - avg_freq_bins_lfp[-1][-1, 0]
