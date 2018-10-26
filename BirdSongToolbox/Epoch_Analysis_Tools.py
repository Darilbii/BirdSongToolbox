"""
Epoch_Analysis_Tools
Tools for analyzing the predictive power of LFP and SUA for predicting Syllable onset
"""

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, lfilter
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider
import ipywidgets as widgets
import os
import scipy.io as sio
import random


# The Following Function Finds the Template for Each Motif for Each Frequency Band on Each Channel
# Edited/Written 2/14/2018

def Mean_match(Features, Sel_Motifs, Num_Chan, Num_Freq, Sn_Len, Gap_Len, OffSet=0):
    ''' Re-Organizes Data for Machine Learning and Visualization. It also Finds the Mean of Each Frequency Band on Each Channel

    Parameters:
    -----------
    Features: list
        Pre-Processed Neural Data
        [Trials]->[Ch]->[ Time(Samples) x Freq (Pass Band)]
    Sel_Motifs: ndarray
        Index of Selected Trials to be Indexed
    Num_Chan:
        Number of Recording Channels
    Num_Freq: int
        Number of Frequency Bands for Each Channel
    Sn_Len: int
        Length (Duration) of avg. Motif for Bird
    Gap_Len: int
        Total Length (Duration) of time Buffer around Trials (To Determine Buffer Before or After Divide by 2)
    OffSet: int
        Number of Samples prior to True Onset of Behavior to Clip Data

    Returns:
    --------
    Channel_Freq_Trials: list
        Re-organized Neural Data
        [Ch]->[Freq]->[Time(Samples) x Trials]
    Channel_Matches: list
        Templates (Mean) of Every Frequency Band for Each Channel
        [Ch]->[Freq]->[Time(Samples) x 1]
    '''
    # Input= shape:(34, 16, 4500, 13)

    Channel_Matches = []  # Index of all Channels
    Channel_Freq_Trials = []

    for Channel in range(Num_Chan):  # Over all Channels
        Matches = []
        Freq_Trials = []

        for Freq_Sel in range(Num_Freq):  # For Range of All Frequency Bins
            Chan_Holder = np.zeros((Sn_Len, 1))  # Initiate Holder for Trials (Motifs) Equal to Length of Motif
            for Motif in Sel_Motifs:  # For Range of Sel Trials
                # Localize to Specified Motif on Specified Frequency Band on Current Channel
                Current_Motif = Features[Motif][Channel][
                                (Gap_Len / 2) - OffSet:((Gap_Len / 2) + Sn_Len) - OffSet, Freq_Sel]  # Variable Gap_Len
                # Line Up all of the Selected Frequencies across All Trials for that Channel
                Chan_Holder = np.column_stack((Chan_Holder, Current_Motif))

            Chan_Holder = np.delete(Chan_Holder, 0, 1)  # Delete the First Column (Initialized Column)
            Freq_Trials.append(Chan_Holder)  # Save all of the Trials for that Frequency on that Channel
            Chan_Means = np.mean(Chan_Holder, axis=1)  # Find Means (Match Filter)
            Chan_StdDevs = np.std(Chan_Holder, axis=1)  ### STD kinda depricated at this point
            Matches.append(
                Chan_Means.reshape(Sn_Len, 1))  # Store all Match Filters for Every Frequency for that Channel

        Channel_Matches.append(Matches)  # Save all of the Templates of Frequencies (Means)
        Channel_Freq_Trials.append(Freq_Trials)  # Save all of the Trials for all Frequencies on each Channel
    return Channel_Freq_Trials, Channel_Matches  # Basic for Function for Vizualizing how seperable the Features are


# Needs to be made into a more Flexible GUI


def Get_LFP_Templates(Trials, Tr_Len, Gap_Len,  Buffer):
    '''Function grabs the time segment of Neural Activity during the designated time Vocal Behavior

    Steps:
    ------
        [1] Iterate over each Trial concatenating each Frequency Band to a desginated Numpy Array
        [2]
        [3]

    Inputs:
    -------
    Trials: list
        List of all Trials of Neural Data
        [Trial]->[ch]->[Song Length (Samples) x Freq. Bin]

    Tr_Len: int

    Gap_Len: int

    Buffer: int


    Returns:
    --------

    '''


    # for Channels in Num_Channels:
    #
    #     for Freq_Band in Num_Freq_Bands:
    #
    #         for Instance in Trials:
    #             # Grab the Specified Trials
    pass

    return

#################################
# Function for Variably clipping Syllables for Machine Learning
# *** Check to Make sure the -1 in Select Motif stage is still accurate with current indexing ***

def Numel(Index):
    Count = 0
    for i in range(len(Index)):
        Count = Count + (len(Index[i]))
    return Count


def Numbad(Index, Offset=int, Tr_Length=int):
    Count = 0
    for i in range(len(Index)):
        Count = Count + len([x for x in range(len(Index[i])) if Index[i][x] < (Offset + Tr_Length)])
    return Count


def Numbad2(Index, ClipLen, Offset=int):
    Count = 0
    for i in range(len(Index)):
        Count = Count + len([x for x in range(len(Index[i])) if Index[i][x] - Offset > ClipLen])
    return Count


def Dyn_LFP_Clipper(Features, Starts, Offset=int, Tr_Length=int):
    ''' This Function Dynamically clips Neural data attributed to a selected label and organizes
    them for future steps. It iterates over EACH full trial clipping ALL examples of ONE label in each trial.
    It should be run repeatedly for clipping all of the designated labels.

    Its output can later be broken into an Initial Training and Final Test Set.


    Starts is a list of Lists containing the Start Times of only One type of Label in each Clipping.
    Also Note that the Starts Argument must be converted to the 1 KHz Sampling Frequency

    Offset = How much prior or after onset'''
    ### Consider removing the Create_Bands Step and consider using len()
    ### ESPECIALLY IF YOU CHANGE THE BANDING CODE

    D = len(Features[:])  # Number of Channels
    F = len(Features[0][:])  # Num of Frequency Bands
    NT = len(Features[0][0][0, :])  # Number of Trials of Dynam. Clipped Training Set
    NEl = Numel(Starts) - Numbad(Starts, Offset=Offset, Tr_Length=Tr_Length) - Numbad2(Starts,
                                                                                       len(Features[0][0][:, 0]),
                                                                                       Offset=Offset)  # Number of Examples

    Dynamic_Templates = []  # Index of all Channels
    Dynamic_Freq_Trials = []
    for Channel in range(0, D):  # Over all Channels
        Matches = []
        Freq_Trials = []
        for l in range(0, F):  # For Range of All Frequency Bins
            Chan_Holder = np.zeros((Tr_Length, NEl))  # Initiate Holder for Trials (Motifs)
            Chan = Features[Channel - 1]  # Select Channel ##### Need to Change Channel to Channel Index (For For Loop)
            Freq = Chan[l]
            Counter = 0  # For stackin all examples of label in full trial
            for Trials in range(NT):
                for Ex in range(len(Starts[Trials])):
                    if Starts[Trials][Ex] - Offset - Tr_Length >= 0 and Starts[Trials][Ex] - Offset < len(Freq):
                        if len(Freq[Starts[Trials][Ex] - Offset - Tr_Length:Starts[Trials][Ex] - Offset, Trials]) == 9:
                            print(Starts[Trials][Ex] - Offset - Tr_Length)
                            print(Starts[Trials][Ex] - Offset)  # Select Motif)
                        Chan_Holder[:, Counter] = Freq[
                                                  Starts[Trials][Ex] - Offset - Tr_Length:Starts[Trials][Ex] - Offset,
                                                  Trials]  # Select Motif
                        Counter = Counter + 1
            Freq_Trials.append(Chan_Holder)  # Save all of the Trials for that Frequency on that Channel
            Chan_Means = np.mean(Chan_Holder, axis=1)  # Find Means (Match Filter)
            Matches.append(
                Chan_Means.reshape(Tr_Length, 1))  # Store all Match Filters for Every Frequency for that Channel
        Dynamic_Templates.append(Matches)
        Dynamic_Freq_Trials.append(Freq_Trials)  # Save all of the Trials for all Frequencies on each Channel

    return Dynamic_Freq_Trials, Dynamic_Templates