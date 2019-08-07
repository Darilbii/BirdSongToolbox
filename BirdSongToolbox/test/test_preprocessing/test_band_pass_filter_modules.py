"""Test functions for the ImportClass """

from BirdSongToolbox.ImportClass import Import_PrePd_Data
from BirdSongToolbox.PreProcessClass import BPF_Master, BPF_Module, Skip_BPF_Module, skip_bpf_master, RR_Neural_Module, RR_Neural_Master
import numpy as np
import pytest



bird_id = 'z020'
date = 'day-2016-06-02'


@pytest.mark.run(order=1)
def test_bpf_module(PrePd_data_dir_path):
    """ Test the bpf_module function"""
    #TODO: Make this more than a smoke test

    PreP_Data = Import_PrePd_Data(bird_id, date, location=PrePd_data_dir_path)

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
def test_bpf_masterPrePd_data_dir_path(PrePd_data_dir_path):
    """test the bpf_master function"""

    #TODO: Make this more than a smoke test

    PreP_Data = Import_PrePd_Data(bird_id, date, location=PrePd_data_dir_path)

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
def test_skip_bpf_module(PrePd_data_dir_path):
    """ test the skip_bpf_module"""
    # TODO: Make this more than a smoke test

    PreP_Data = Import_PrePd_Data(bird_id, date, location=PrePd_data_dir_path)

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
def test_skip_bpf_master(PrePd_data_dir_path):
    """ test the skip_bpf_module"""
    # TODO: Make this more than a smoke test

    PreP_Data = Import_PrePd_Data(bird_id, date, location=PrePd_data_dir_path)

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
