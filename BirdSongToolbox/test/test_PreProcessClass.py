"""Test functions for the ImportClass """

from BirdSongToolbox.ImportClass import Import_PrePd_Data
from BirdSongToolbox.PreProcessClass import BPF_Master, BPF_Module, Skip_BPF_Module, skip_bpf_master, RR_Neural_Module, RR_Neural_Master
from BirdSongToolbox.config.settings import TEST_DATA_DIR
from BirdSongToolbox.PreProcTools import Good_Channel_Index
import numpy as np
import pytest



bird_id = 'z020'
date = 'day-2016-06-02'


@pytest.mark.run(order=1)
def test_bpf_module():
    """ Test the bpf_module function"""
    #TODO: Make this more than a smoke test

    PreP_Data = Import_PrePd_Data(bird_id, date, location=TEST_DATA_DIR)

    Channels = PreP_Data.Song_Neural[0]
    Freq_Bands = ([10], [1])
    SN_L = PreP_Data.Sn_Len
    Gp_L = PreP_Data.Gap_Len
    Num_Chan = PreP_Data. Num_Chan
    order_num = 175
    fs = PreP_Data.Fs
    FiltFilt = True

    song_length, number_channels = np.shape(Channels)
    tops, bottoms = Freq_Bands

    freq_bins = BPF_Module(Channels, Freq_Bands=Freq_Bands, SN_L=SN_L, Gp_L=Gp_L, Num_Chan=Num_Chan, Num_Freq=len(tops), order_num=order_num, fs=fs,FiltFilt=FiltFilt)



    # Smoke Tests
    assert isinstance(freq_bins, list)
    assert isinstance(freq_bins[0], np.ndarray)
    assert len(freq_bins) == number_channels
    assert (song_length, len(tops)) == np.shape(freq_bins[0])


@pytest.mark.run(order=1)
def test_bpf_master():
    """test the bpf_master function"""

    #TODO: Make this more than a smoke test

    PreP_Data = Import_PrePd_Data(bird_id, date, location=TEST_DATA_DIR)

    Channels = PreP_Data.Song_Neural
    Num_Trials = PreP_Data.Num_Motifs
    Freq_Bands = ([10], [1])
    SN_L = PreP_Data.Sn_Len
    Gp_L = PreP_Data.Gap_Len
    Num_Chan = PreP_Data. Num_Chan
    order_num = 175
    fs = PreP_Data.Fs
    FiltFilt = True

    song_length, number_channels = np.shape(Channels[0])
    assert number_channels == Num_Chan
    tops, bottoms = Freq_Bands

    bpf_motifs = BPF_Master(Channels, Num_Trials = Num_Trials, Freq_Bands=Freq_Bands, SN_L=SN_L, Gp_L=Gp_L,
                            Num_Chan=Num_Chan, Num_Freq=len(tops), order_num=order_num, fs=fs, FiltFilt=FiltFilt,
                            verbose=False)

    # Smoke Test

    assert isinstance(bpf_motifs, list)
    assert isinstance(bpf_motifs[0], list)
    assert isinstance(bpf_motifs[0][0], np.ndarray)
    assert len(bpf_motifs) == Num_Trials
    assert len(bpf_motifs[0]) == Num_Chan
    assert np.shape(bpf_motifs[0][0]) == (song_length, len(tops))


@pytest.mark.run(order=1)
def test_skip_bpf_module():
    """ test the skip_bpf_module"""
    # TODO: Make this more than a smoke test

    PreP_Data = Import_PrePd_Data(bird_id, date, location=TEST_DATA_DIR)

    Channels = PreP_Data.Song_Neural[0]
    SN_L = PreP_Data.Sn_Len
    Gp_L = PreP_Data.Gap_Len
    Num_Chan = PreP_Data.Num_Chan



    song_length, number_channels = np.shape(Channels)
    assert number_channels == Num_Chan

    skip_bpf_motifs = Skip_BPF_Module(Channels, SN_L=SN_L, Gp_L=Gp_L, Num_Chan=Num_Chan)

    # Smoke Tests

    assert isinstance(skip_bpf_motifs, list)
    assert isinstance(skip_bpf_motifs[0], np.ndarray)
    assert len(skip_bpf_motifs) == number_channels
    assert np.shape(skip_bpf_motifs[0]) == (song_length, 1)


    # Unit Test
    assert skip_bpf_motifs[0][0, 0] == Channels[0, 0]
    assert skip_bpf_motifs[0][-1, 0] == Channels[-1, 0]
    assert skip_bpf_motifs[-1][0, 0] == Channels[0, -1]
    assert skip_bpf_motifs[-1][-1, 0] == Channels[-1, -1]


@pytest.mark.run(order=1)
def test_skip_bpf_master():
    """ test the skip_bpf_module"""
    # TODO: Make this more than a smoke test

    PreP_Data = Import_PrePd_Data(bird_id, date, location=TEST_DATA_DIR)

    Channels = PreP_Data.Song_Neural
    SN_L = PreP_Data.Sn_Len
    Gp_L = PreP_Data.Gap_Len
    Num_Chan = PreP_Data.Num_Chan
    Num_Trials = PreP_Data.Num_Motifs

    song_length, number_channels = np.shape(Channels[0])
    assert number_channels == Num_Chan

    skip_bpf_motifs = skip_bpf_master(Channels, SN_L=SN_L, Gp_L=Gp_L, Num_Chan=Num_Chan)

    # Smoke Tests

    assert isinstance(skip_bpf_motifs, list)
    assert isinstance(skip_bpf_motifs[0], list)
    assert isinstance(skip_bpf_motifs[0][0], np.ndarray)
    assert len(skip_bpf_motifs) == Num_Trials
    assert len(skip_bpf_motifs[0]) == Num_Chan
    assert np.shape(skip_bpf_motifs[0][0]) == (song_length, 1)


    # Unit Test
    assert skip_bpf_motifs[0][0][0, 0] == Channels[0][0, 0]
    assert skip_bpf_motifs[0][0][-1, 0] == Channels[0][-1, 0]
    assert skip_bpf_motifs[0][-1][0, 0] == Channels[0][0, -1]
    assert skip_bpf_motifs[0][-1][-1, 0] == Channels[0][-1, -1]

    assert skip_bpf_motifs[-1][0][0, 0] == Channels[-1][0, 0]
    assert skip_bpf_motifs[-1][0][-1, 0] == Channels[-1][-1, 0]
    assert skip_bpf_motifs[-1][-1][0, 0] == Channels[-1][0, -1]
    assert skip_bpf_motifs[-1][-1][-1, 0] == Channels[-1][-1, -1]

    assert np.all(skip_bpf_motifs[-1][0][:, 0] == Channels[-1][:, 0])



#
# @pytest.mark.run(order=1)
# def test_rr_neural_module():
#     PreP_Data = Import_PrePd_Data(bird_id, date, location=TEST_DATA_DIR)
#
#     Channels = PreP_Data.Song_Neural[0]
#     Freq_Bands = ([10], [1])
#     SN_L = PreP_Data.Sn_Len
#     Gp_L = PreP_Data.Gap_Len
#     Num_Chan = PreP_Data.Num_Chan
#     order_num = 175
#     fs = PreP_Data.Fs
#     FiltFilt = True
#     Good_Channels = Good_Channel_Index(PreP_Data.Num_Chan, PreP_Data.Bad_Channels)
#
#     song_length, number_channels = np.shape(Channels)
#     tops, bottoms = Freq_Bands
#     num_freq = len(tops)
#
#     Frequencies = BPF_Module(Channels, Freq_Bands=Freq_Bands, SN_L=SN_L, Gp_L=Gp_L, Num_Chan=Num_Chan,
#                              Num_Freq=num_freq, order_num=order_num, fs=fs, FiltFilt=FiltFilt)
#
#     freq_bins_rr, avg_freq_bins_lfp = RR_Neural_Module(Frequencies, Good_Channels=Good_Channels, SN_L=SN_L, Gp_L=Gp_L,
#                                                        Num_Freq=num_freq)
#
#     # Smoke Tests: freq_bins_rr
#     assert isinstance(freq_bins_rr, list)
#     assert isinstance(freq_bins_rr[0], np.ndarray)
#     assert np.shape(freq_bins_rr) == (number_channels, song_length, num_freq)
#
#     # Smoke Test: avg_freq_nins_lfp
#     assert isinstance(avg_freq_bins_lfp, np.ndarray)
#     assert np.shape(avg_freq_bins_lfp) == (song_length, num_freq)
#
#     # Unit Test: freq_bins_rr
#     assert freq_bins_rr[0][0,0] == Frequencies[0][0,0] - avg_freq_bins_lfp[0,0]
#     assert freq_bins_rr[0][-1,0] == Frequencies[0][-1,0] - avg_freq_bins_lfp[-1,0]
#     assert freq_bins_rr[-1][0,0] == Frequencies[-1][0,0] - avg_freq_bins_lfp[0,0]
#     assert freq_bins_rr[-1][-1,0] == Frequencies[-1][-1,0] - avg_freq_bins_lfp[-1,0]
#
#
# @pytest.mark.run(order=1)
# def test_rr_neural_master():
#     PreP_Data = Import_PrePd_Data(bird_id, date, location=TEST_DATA_DIR)
#
#     Channels = PreP_Data.Song_Neural
#     Freq_Bands = ([10], [1])
#     SN_L = PreP_Data.Sn_Len
#     Gp_L = PreP_Data.Gap_Len
#     Num_Chan = PreP_Data.Num_Chan
#     Num_Trials = PreP_Data.Num_Motifs
#     order_num = 175
#     fs = PreP_Data.Fs
#     FiltFilt = True
#
#     Good_Channels = Good_Channel_Index(PreP_Data.Num_Chan, PreP_Data.Bad_Channels)
#     song_length, number_channels = np.shape(Channels[0])
#     tops, bottoms = Freq_Bands
#     num_freq = len(tops)
#
#     assert number_channels == Num_Chan
#
#     Frequencies = BPF_Master(Channels, Num_Trials=Num_Trials, Freq_Bands=Freq_Bands, SN_L=SN_L, Gp_L=Gp_L,
#                              Num_Chan=Num_Chan, Num_Freq=len(tops), order_num=order_num, fs=fs, FiltFilt=FiltFilt,
#                              verbose=False)
#
#     freq_bins_rr, avg_freq_bins_lfp = RR_Neural_Master(Frequencies, Num_Trials=Num_Trials, Good_Channels=Good_Channels,
#                                                        SN_L=SN_L, Gp_L=Gp_L, Num_Freq=num_freq)
#
#     # Smoke Tests: freq_bins_rr
#     assert isinstance(freq_bins_rr, list)
#     assert isinstance(freq_bins_rr[0], list)
#     assert isinstance(freq_bins_rr[0][0], np.ndarray)
#     assert np.shape(freq_bins_rr) == (Num_Trials, number_channels, song_length, num_freq)
#
#     # Smoke Test: avg_freq_nins_lfp
#     assert isinstance(avg_freq_bins_lfp, list)
#     assert isinstance(avg_freq_bins_lfp[0], np.ndarray)
#     assert np.shape(avg_freq_bins_lfp) == (Num_Trials, song_length, num_freq)
#
#     # Unit Test: freq_bins_rr
#     assert freq_bins_rr[0][0][0, 0] == Frequencies[0][0][0,0 ] - avg_freq_bins_lfp[0][0, 0]
#     assert freq_bins_rr[0][0][-1, 0] == Frequencies[0][0][-1, 0] - avg_freq_bins_lfp[0][-1, 0]
#     assert freq_bins_rr[0][-1][0, 0] == Frequencies[0][-1][0, 0] - avg_freq_bins_lfp[0][0, 0]
#     assert freq_bins_rr[0][-1][-1, 0] == Frequencies[0][-1][-1, 0] - avg_freq_bins_lfp[0][-1, 0]
#     assert freq_bins_rr[-1][0][0, 0] == Frequencies[-1][0][0,0 ] - avg_freq_bins_lfp[-1][0, 0]
#     assert freq_bins_rr[-1][0][-1, 0] == Frequencies[-1][0][-1, 0] - avg_freq_bins_lfp[-1][-1, 0]
#     assert freq_bins_rr[-1][-1][0, 0] == Frequencies[-1][-1][0, 0] - avg_freq_bins_lfp[-1][0, 0]
#     assert freq_bins_rr[-1][-1][-1, 0] == Frequencies[-1][-1][-1, 0] - avg_freq_bins_lfp[-1][-1, 0]








