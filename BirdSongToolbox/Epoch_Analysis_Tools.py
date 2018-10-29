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

    ** Limit: This code can only parse the central motif of the Epoch **

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
            # Chan_StdDevs = np.std(Chan_Holder, axis=1)  ### STD kinda depricated at this point
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

################################################################################################################################################################################################
## First Function for Offline Epoch Predictions

# This function is the start of a way to dynamically clip trials for realtime classification code.
# For Zeke


## The Following Function Finds the Template for Each Motif for Each Frequency Band on Each Channel
## ***Structural Detail Notes were made on 4/27/17***
##This was copied over from Feature Classification...(Development)
############ IT HAS SINCE BEEN CHANGED From Original ############
# Added Functionality for Handling Sliding Frequency Band (5/23/2017)
# This is a Altered Version of Mean_Match. It Intakes the Full PrePd Motif Trials (Including Gaps)

def Full_Trial_LFP_Clipper_Old(Features, Sel_Motifs, SS=15, Low=5, Sel_Feature=2, Sliding=False, SUPP=False):
    ''' Grabs every epoch in Sel_Motifs for Each Frequency Band on Each Channel and returns them in a structure list


    Information:
    ------------
        ** This Function must be run on the Train and Test Set Individually prior to continuing through the analysis**

    :param Features:
    :param Sel_Motifs:
    :param SS:
    :param Low:
    :param Sel_Feature:
    :param Sliding:
    :param SUPP:
    :return:

    Returns:
    --------
     Channel_Full_Freq_Trials: list
        [Ch]->[Freq]->(Time Samples x Trials)

    '''
    # Remove:  Sel_Freq, Channel = 1, SN = 1,
    # Offset = How much prior or after onset

    TOP, BOTTOM = Create_Bands(StepSize=SS, Lowest=Low, Slide=Sliding,
                               Suppress=SUPP)  # Run Function to create variables for the Frequency Bands

    # 2: Initiate Variables

    Selected_Feature_Type = []  # Holder for Selected Feature Type (from Sel_Feature)

    #     Chan = Features[Channel - 1]
    # Selected_Feature_Type = Features[Sel_Feature]
    #
    # B = len(Features[Sel_Feature][0][0][:, 0])  # Length of full Trial
    # D = len(Features[Sel_Feature][0][:])

    Channel_Matches = []  # Index of all Channels
    Channel_Freq_Trials = []
    Channel_Full_Freq_Trials = []

    for Channel in range(0, D):  # Over all Channels
        Freq_Trials = []
        Freq_Full_Trials = []
        for l in range(0, len(TOP)):  # For Range of All Frequency Bins
            Chan_Full_Holder = np.zeros((4500, 1))  # Initiate Holder for Trials (Motifs)
            for motif in Sel_Motifs:  # For each value of Sel_Motifs
                # MOTIF = Selected_Feature_Type[motif - 1]  # Select Motif
                # Chan = MOTIF[Channel - 1]  # Select Channel ##### Need to Change Channel to Channel Index (For For Loop)
                # Current_Full_Motif = Chan[:, l]  # Select Motif
                Current_Full_Motif = Selected_Feature_Type[motif - 1][Channel - 1][:, l]  # Select[Motif][Ch][Epoch, Freq]

                # Line Up all of the Selected Frequencies across All Trials for that Channel
                Chan_Full_Holder = np.column_stack((Chan_Full_Holder, Current_Full_Motif))
            Chan_Full_Holder = np.delete(Chan_Full_Holder, 0, 1)  # Delete the First Column (Initialized Column)
            Freq_Full_Trials.append(Chan_Full_Holder)  # Save all of the Trials for that Frequency on that Channel
        Channel_Full_Freq_Trials.append(Freq_Full_Trials)

    return Channel_Full_Freq_Trials


def Full_Trial_LFP_Clipper(Neural, Sel_Motifs, Num_Freq, Num_Chan, Sn_Len, Gap_Len):
    """Grabs every epoch in Sel_Motifs for Each Frequency Band on Each Channel and returns them in a structure list

    Information:
    ------------
        ** This Function must be run on the Train and Test Set Individually prior to continuing through the analysis**

    Parameters:
    -----------
    Neural:
        [Number of Trials]-> [Ch] -> [Trial Length (Samples @ User Designated Sample Rate) x Freq_Bands]
    Sel_Motifs:m list
        List of Epochs to be used based on Designated Label
    Num_Freq: int
        Number of Band Passed Filtered Signals
    Num_Chan: int
        Number of Recording Channels
    Sn_Len: int
        Stereotyped Length (In Samples) of Motif
    Gap_Len: int
        Length of Time Buffer Before and After Motif

    Returns:
    --------
     Channel_Full_Freq_Trials: list
        [Ch]->[Freq]->(Time Samples x Trials)

    """

    # 1: Initiate Variables

    Channel_Full_Freq_Trials = []

    for Channel in range(Num_Chan):  # Over all Channels
        Freq_Full_Trials = []
        for Freq in range(Num_Freq):  # For Range of All Frequency Bins
            Chan_Full_Holder = np.zeros((Sn_Len + Gap_Len, 1))  # Initiate Holder for Trials (Motifs)
            for motif in Sel_Motifs:  # For each value of Sel_Motifs
                Current_Full_Motif = Neural[motif - 1][Channel][:, Freq]  # Select[Motif][Ch][Epoch, Freq]

                # Line Up all of the Selected Frequencies across All Trials for that Channel
                Chan_Full_Holder = np.column_stack((Chan_Full_Holder, Current_Full_Motif))
            Chan_Full_Holder = np.delete(Chan_Full_Holder, 0, 1)  # Delete the First Column (Initialized Column)
            Freq_Full_Trials.append(Chan_Full_Holder)  # Save all of the Trials for that Frequency on that Channel
        Channel_Full_Freq_Trials.append(Freq_Full_Trials)

    return Channel_Full_Freq_Trials


########################################################################################################################
## Label Handling Functions
# Function to Focus on only one type of Label

### Added int() to pipeline on 8/19

def Label_Focus(Focus, Labels, Starts):
    """ Create a list of every instance of the User defined User Label

    Parameters:
    -----------
    Focus: str or int
        User defined Label to focus on
    Labels:

    Starts:


    Returns:
    --------
    Label_Index: list
        List of all start frames of every instances of the label of focus
        [Num_Trials]->[Num_Exs]
    """
    Label_Index = []

    for i in range(len(Labels)):
        Trial_Labels = [int(Starts[i][x] / 30) for x in range(len(Labels[i])) if Labels[i][x] == Focus]
        Label_Index.append(Trial_Labels)
    return Label_Index


# Function for Grouping Multiple Labels into 1 Label (e.g. Combine Calls and Introductory Notes)

def Label_Grouper(Focuses, Labels, Starts):
    """Group Selected Labels together into One Label
            e.g. Combine Calls and Introductory Notes"""
    Label_Index = []

    for i in range(len(Labels)):
        Group_Labels = []
        for j in range(len(Focuses)):
            Trial_Labels = [int(Starts[i][x] / 30) for x in range(len(Labels[i])) if Labels[i][x] == Focuses[j]]
            Group_Labels.extend(Trial_Labels)
        Label_Index.append(Group_Labels)
    return Label_Index

# Function for grabing more examples from a onset

def Slider(Ext_Starts, Slide=int, Step=False):
    """
    Parameters:
    -----------
    Ext_Starts: list

    Slide: int (optional)

    Step: bool (optional)
        (defaults to False)
    Return:
    -------

    """
    Num_Trials = len(Ext_Starts)

    Slid_starts = []
    for i in range(len(Ext_Starts)):
        Slid_Trial = []
        for j in range(len(Ext_Starts[i])):
            if Step == False:
                for k in range(Slide):
                    Slid_Trial.append(Ext_Starts[i][j] + k)
            if Step == True:
                for k in range(0, Slide, Step):
                    Slid_Trial.append(Ext_Starts[i][j] + k)
        Slid_starts.append(Slid_Trial)
    return Slid_starts


def Label_Extract_Pipeline(Full_Trials, All_Labels, Time_Stamps, Label_Instructions, Offset=int, Tr_Length=int,
                           Slide=None, Step=False):
    """Extracts all of the Neural Data Examples of User Selected Labels and return them in the designated manner.

    Label_Instructions = tells the Function what labels to extract and whether to group them together

    Parameters:
    -----------

    Returns:
    -------
    clippings:

    templates:
    """

    clippings = []
    templates = []

    for i in range(len(Label_Instructions)):
        if type(Label_Instructions[i]) == int or type(Label_Instructions[i]) == str:
            label_starts = Label_Focus(Label_Instructions[i], All_Labels, Time_Stamps)
        else:
            label_starts = Label_Grouper(Label_Instructions[i], All_Labels, Time_Stamps)

        if type(Slide) == int:
            label_starts = Slider(label_starts, Slide=Slide, Step=Step)

        clips, temps = Dyn_LFP_Clipper(Full_Trials, label_starts, Offset=Offset, Tr_Length=Tr_Length)
        clippings.append(clips)
        templates.append(temps)
    return clippings, templates


def Power_Extraction(Clipped_Trials):
    """

    :param Clipped_Trials:
    :return:
    """
    Extracted_Power = []
    for i in range(len(Clipped_Trials)):
        Extracted_Power.append(Find_Power(Clipped_Trials[i]))
    return Extracted_Power



#TODO: Need to change this to doing:  (RMS, Log_RMS, MS) Consider Log Scale

def Find_Power(Features, Pow_Method='Basic'):
    """ Function to Find the Power for all Trials (Intermediate Preprocessing Step)

    Features: list
        Structure:
    Pow_Method: str
        Method by which power is taken (Options: 'Basic': Mean , 'MS': Mean Squared, 'RMS': Root Mean Square)
    :return:
    """
    # Create Variable for IndexingF
    num_trials = len(Features[0][0][0, :])  # Number of Trials of Dynam. Clipped Training Set

    # Create Lists
    Power_Trials = []

    for Channel in range(len(Features[:])):  # Over all Channels
        Freq_Trials = []
        for l in range(len(Features[0][:])):  # For Range of All Frequency Bins
            #             print abs(Features[Channel - 1][l])
            if Pow_Method == 'Basic':
                chan_holder = np.average(abs(Features[Channel][l]), axis=0)
            if Pow_Method == 'MS':
                chan_holder = np.mean(np.power(Features[Channel][l], 2), axis=0)
            if Pow_Method == 'RMS':
                chan_holder = np.power(np.mean(np.power(Features[Channel][l], 2), axis=0), .5)

            chan_holder = np.reshape(chan_holder, (num_trials, 1))
            Freq_Trials.append(chan_holder)  # Save all of the Trials for that Frequency on that Channel
        Power_Trials.append(Freq_Trials)  # Save all of the Trials for all Frequencies on each Channel
    return Power_Trials


def ML_Order_Pipeline(Extracted_Features):
    """

    :param Extracted_Features:
    :return:
    """
    ML_Ready = np.zeros((1, (len(Extracted_Features[0]) * len(Extracted_Features[0][0]))))
    ML_Labels = np.zeros((1, 1))
    for i in range(len(Extracted_Features)):
        Ordered_Trials, Ordered_Index = ML_Order(Extracted_Features[i])
        ML_Ready = np.concatenate((ML_Ready, Ordered_Trials), axis=0)

        # Handels Labels so they are flexible when grouping
        ROW, COLL = np.shape(Ordered_Trials)
        Dyn_Labels = np.zeros([ROW, 1])
        Dyn_Labels[:, 0] = i
        ML_Labels = np.concatenate((ML_Labels, Dyn_Labels), axis=0)

    ML_Ready = np.delete(ML_Ready, 0, 0)
    ML_Labels = np.delete(ML_Labels, 0, 0)
    return ML_Ready, ML_Labels, Ordered_Index

########################################################################################################################
# Function for Variably clipping Syllables for Machine Learning
# *** Check to Make sure the -1 in Select Motif stage is still accurate with current indexing ***

def Numel(Index):
    """Get the Number of Elements"""
    Count = 0
    for i in range(len(Index)):
        Count = Count + (len(Index[i]))
    return Count


def Numbad(Index, Offset=int, Tr_Length=int):
    """ Returns the number of Label instances that finish after the end of the Epoch (Prevents Index errors propagating)

    Parameters:
    ----------
    Index: list
        A list of Lists containing the Start Times of only One type of Label in each Clipping.
        Also Note that the Starts Argument must be converted to the 1 KHz Sampling Frequency

    Offset: int

    Tr_Length: int

    return:
    -------
    count: int
        number of bad indexes
    """
    Count = 0
    for i in range(len(Index)):
        Count = Count + len([x for x in range(len(Index[i])) if Index[i][x] < (Offset + Tr_Length)])
    return Count


def Numbad2(Index, ClipLen, Offset=int):
    """Returns the number of Label instances that start before the start of the Epoch (Prevents Index errors propagating)

    Parameters:
    ----------
    Index: list
        A list of Lists containing the Start Times of only One type of Label in each Clipping.
        Also Note that the Starts Argument must be converted to the 1 KHz Sampling Frequency

    ClipLen: int

    Offset: int

    return:
    -------
    count: int
        number of bad indexes
    """
    Count = 0
    for i in range(len(Index)):
        Count = Count + len([x for x in range(len(Index[i])) if Index[i][x] - Offset > ClipLen])
    return Count


def Dyn_LFP_Clipper_Old(Features, Starts, Offset=int, Tr_Length=int):
    """This Function Dynamically clips Neural data prior to a selected label and re-organizes them for future ML use.
    (Prior Neural Data Only)

    Information:
    ------------
        It iterates over EACH Epoch clipping ALL examples of ONE label in each trial.
        It should be run repeatedly for clipping all of the designated labels.

        Its output can later be broken into an Initial Training and Final Test Set.

    Parameters:
    -----------
    Starts: list
        A list of Lists containing the Start Times of only One type of Label in each Clipping.
    Also Note that the Starts Argument must be converted to the 1 KHz Sampling Frequency

    Offset = How much prior or after onset

    Returns:
    --------

    """

    ### Consider removing the Create_Bands Step and consider using len()
    ### ESPECIALLY IF YOU CHANGE THE BANDING CODE

    D = len(Features[:])  # Number of Channels
    F = len(Features[0][:])  # Num of Frequency Bands
    NT = len(Features[0][0][0, :])  # Number of Trials of Dynam. Clipped Training Set
    NEl = Numel(Starts) - Numbad(Starts, Offset=Offset, Tr_Length=Tr_Length) - Numbad2(Starts, len(Features[0][0][:, 0]), Offset=Offset)  # Number of Examples

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
            Matches.append(Chan_Means.reshape(Tr_Length, 1))  # Store all Match Filters for Every Frequency for that Channel
        Dynamic_Templates.append(Matches)
        Dynamic_Freq_Trials.append(Freq_Trials)  # Save all of the Trials for all Frequencies on each Channel

    return Dynamic_Freq_Trials, Dynamic_Templates


def Dyn_LFP_Clipper(Features: list, Starts, Offset=int, Tr_Length=int):
    """This Function Dynamically clips Neural data prior to a selected label and re-organizes them for future use.

    Information:
    ------------
        **This function assumes that the Data has Been limited to only Epoch that will be Trained or Tested**
        It iterates over EACH Epoch clipping ALL examples of ONE label in each trial.
        It should be run repeatedly for clipping all of the designated labels.

        Its output can later be broken into an Initial Training and Final Test Set. [May Not Be True]

    Parameters:
    -----------
    Features: list
        [Ch]->[Freq]->(Time Samples x Trials)

    Starts: list
        A list of Lists containing the Start Times of only One type of Label in each Clipping.
    Also Note that the Starts Argument must be converted to the 1 KHz Sampling Frequency

    Offset = How much prior or after onset

    Returns:
    --------
    Dynamic_Freq_Trials: list
        List of Stacked Neural Data that corresponds to the Label designated
        [Ch]->[Freq]->(Time (Samples) x Examples of Labels)
        Note: The Number of Examples of Label does not always equal the total number of examples total as some push pass
            the timeframe of the Epoch and are excluded
    Dynamic_Templates: list
        List of Stacked Template Neural Data that corresponds to the Label designated (Templates are the mean of trials)
        [Ch]->[Freq]->(Time (Samples) x Examples of Labels)
        Note: The Number of Examples of Label does not always equal the total number of examples total as some push past
            the time frame of the Epoch and are excluded
    """

    ### Consider removing the Create_Bands Step and consider using len()
    ### ESPECIALLY IF YOU CHANGE THE BANDING CODE

    num_chan = len(Features[:])  # Number of Channels
    freq_bands = len(Features[0][:])  # Num of Frequency Bands
    num_trials = len(Features[0][0][0, :])  # Number of Trials of Dynam. Clipped Training Set
    num_examples = Numel(Starts) - Numbad(Starts, Offset=Offset, Tr_Length=Tr_Length) - Numbad2(Starts, len(Features[0][0][:, 0]), Offset=Offset)  # Number of Examples

    Dynamic_Templates = []  # Index of all Channels
    Dynamic_Freq_Trials = []
    for Channel in range(num_chan):  # Over all Channels
        Matches = []
        Freq_Trials = []
        for l in range(0, freq_bands):  # For Range of All Frequency Bins
            Chan_Holder = np.zeros((Tr_Length, num_examples))  # Initiate Holder for Trials (Motifs)
            sel_freq_epochs = Features[Channel][l]
            Counter = 0  # For stacking all examples of label in full trial
            for epoch in range(num_trials):
                for example in range(len(Starts[epoch])):
                    if Starts[epoch][example] - Offset - Tr_Length >= 0 and Starts[epoch][example] - Offset < len(sel_freq_epochs):
                        # if len(sel_freq_epochs[Starts[epoch][example] - Offset - Tr_Length:Starts[epoch][example] - Offset, epoch]) == 9:
                        #     print(Starts[epoch][example] - Offset - Tr_Length)
                        #     print(Starts[epoch][example] - Offset)  # Select Motif)
                        Chan_Holder[:, Counter] = sel_freq_epochs[Starts[epoch][example] - Offset - Tr_Length:Starts[epoch][example] - Offset, epoch]  # Select Motif
                        Counter = Counter + 1
            Freq_Trials.append(Chan_Holder)  # Save all of the Trials for that Frequency on that Channel
            Chan_Means = np.mean(Chan_Holder, axis=1)  # Find Means (Match Filter)
            Matches.append(Chan_Means.reshape(Tr_Length, 1))  # Store all Match Filters for Every Frequency for that Channel
        Dynamic_Templates.append(Matches)
        Dynamic_Freq_Trials.append(Freq_Trials)  # Save all of the Trials for all Frequencies on each Channel

    return Dynamic_Freq_Trials, Dynamic_Templates

########################################################################################################################
