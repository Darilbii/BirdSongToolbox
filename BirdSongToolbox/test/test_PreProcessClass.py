"""Test functions for the ImportClass """

from BirdSongToolbox.ImportClass import Import_PrePd_Data
from BirdSongToolbox.PreProcessClass import *
import numpy as np

def test_all_modules():
    """This is a holder for testing all of the Modules until I implement a better approach to test data"""

    bird_id = 'z020'
    date = 'day-2016-06-02'

    PreP_Data = Import_PrePd_Data(bird_id, date)

    def test_bpf_module():
        """ Test the bpf_module function"""
        #TODO: Make this more than a smoke test

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

    def test_bpf_master():
        """test the bpf_master function"""

        #TODO: Make this more than a smoke test

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

    test_bpf_module()
    test_bpf_master()



