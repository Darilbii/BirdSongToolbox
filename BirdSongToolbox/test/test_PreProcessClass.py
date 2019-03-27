"""Test functions for the ImportClass """

from BirdSongToolbox.ImportClass import Import_PrePd_Data
from BirdSongToolbox.PreProcessClass import BPF_Master, BPF_Module, Skip_BPF_Module, RR_Neural_Module, RR_Neural_Master
from BirdSongToolbox.config.settings import TEST_DATA_DIR
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
    assert (song_length, 1) == np.shape(skip_bpf_motifs[0])
    assert len(skip_bpf_motifs[0]) == len(Channels[:, 0])

    # Unit Test
    assert Channels[0, 0] == skip_bpf_motifs[0][0, 0]
    assert Channels[-1, 0] == skip_bpf_motifs[0][-1, 0]
    assert Channels[0, -1] == skip_bpf_motifs[-1][0, 0]
    assert Channels[-1, -1] == skip_bpf_motifs[-1][-1, 0]


# @pytest.mark.run(order=1)
# def test_rr_neural_module():
#     PreP_Data = Import_PrePd_Data(bird_id, date, location=TEST_DATA_DIR)
#
#     Frequencies = PreP_Data.Song_Neural[0]
#     Freq_Bands = ([10], [1])
#     SN_L = PreP_Data.Sn_Len
#     Gp_L = PreP_Data.Gap_Len
#     Num_Chan = PreP_Data.Num_Chan
#     order_num = 175
#     fs = PreP_Data.Fs
#     FiltFilt = True
#
#     song_length, number_channels = np.shape(Channels)
#     tops, bottoms = Freq_Bands
#
#     freq_bins = RR_Neural_Module(Frequencies, Freq_Bands=Freq_Bands, SN_L=SN_L, Gp_L=Gp_L, Num_Chan=Num_Chan, Num_Freq=len(tops),
#                            order_num=order_num, fs=fs, FiltFilt=FiltFilt)





