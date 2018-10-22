#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

import random


# These Commands implement a method Aashish used

def bandpass(lowcut, highcut, fs, order=203):
    ''' Design FIR Bandpass Filter
    Parameters:
    -----------
    lowcut: int
        Low Frequency cut-off of Filter
    highcut: int
        High Frequency cut-off of Filter
    order: int (Optional)
        Order of Filter to be created (Defaults to 203)

    Returns:
    --------
    b: ndarray
        Coefficients of numerator for Filter (Produced by scipy.signal.firwin)

    a: int
        Filter Denominator's Coefficients (Defaults to 1)
    '''

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    a = 1 # filter denominator coeffs
    b = scipy.signal.firwin(order, [low, high], None, 'hann', pass_zero=False) # 203 order
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order_num=204):
    ''' Bandpass Filter input data using FIR Filter created using parameters passed. The Filter is used twice using the FiltFilt command to remove phase distortion

    Parameters:
    -----------
    data: list
        Data to be bandpass filtered
    lowcut: int
        Low Frequency cut-off of Filter
    highcut: int
        High Frequency cut-off of Filter
    fs: int
        Sampled Frequency of data to be filtered

    order: int (Optional)
        Order of Filter to be created (Defaults to 203)

    Returns:
    --------
    y: List
        Bandpass Filtered Data
    '''

    b, a = bandpass(lowcut, highcut, fs, order=order_num)

    y = scipy.signal.filtfilt(b, a, data)
    return y

def bandpass_filter_causal(data, lowcut, highcut, fs, order_num=204):
    ''' Bandpass Filter input data using FIR Filter created using parameters passed (Calls Func.: bandpass()) The Filter is passed once to show Causal Filtered Data

    Parameters:
    -----------
    data: list
        Data to be bandpass filtered
    lowcut: int
        Low Frequency cut-off of Filter
    highcut: int
        High Frequency cut-off of Filter
    fs: int
        Sampled Frequency of data to be filtered

    order: int (Optional)
        Order of Filter to be created (Defaults to 203)

    Returns:
    --------
    y: List
        Bandpass Filtered Data
    '''

    Correct_order_num = order_num *2
    b, a = bandpass(lowcut, highcut, fs, order=Correct_order_num)

    y = scipy.signal.lfilter(b, a, data)
    return y


# Creates Index for Frequency Band Boundaries
# Additional Functionality Added= Sliding Set Band Width [5/23/2017]

def Create_Bands(StepSize=20, Lowest=0, Slide=False, Suppress=False):
    ''' Creates Index for Frequency Pass Band Boundaries (High and Low Cuttoff Frequencies)
    Parameters:
    -----------
    StepSize: int
        Width of All Bandpass Filters
    Lowest: int
        Lowest frequency to start
    Slide: bool (Optional)
        If True Bandpass Filters will have a stepsize of 1 Hz (Defaults to False)
    Suppress: bool (Optional)
        If True Function's print statements will be ignored (Defaults to False) [Helps to reduce unnecesary printing steps]

    Returns:
    --------
    Top: list
        List of High Frequency Cuttoffs of Bandpass's used
    Bottom: list
        List of Low Frequency Cutoffs
    '''
    Top = []  # For holding the Top of all the pass bands
    Bottom = []  # For holding the Bottom of all the pass bands

    if Slide == False:
        binwidths = np.arange(Lowest, 200, StepSize)  # Create Np Array of Bin Starts (Bottoms)

    if Slide == True:
        binwidths = np.arange(Lowest, 201 - StepSize, 1)  # Create Np Array of Bin Starts (Bottoms)

    B = len(binwidths)  # Number of Frequency Bins

    Bottom = binwidths  # np array of Bottoms of all the pass bands
    Top = Bottom + StepSize  # np array of Tops of all the pass bands

    ## The Following 5 lines are for Visual Sanity Checks
    if Suppress == False:
        print 'Number of Bins:' + str(B)

        print 'Roofs:'
        print Top
        print 'Floors:'
        print Bottom
    return Top, Bottom


# Function to Band Pass Selected Frequency Bands
# Additional Functionality Added= Sliding Set Band Width [5/23/2017]
# Additioanl Functionality Added= Dynamic Treatment of Channel Number [1/22/2018]

def Sliding_BPF(Channels, SN_L=int, Gp_L=int, StepSize=20, Lowest=0, Order=175, fs=1000, FiltFilt=True, Slide=False,
                SUPP=False):
    '''Consecutive Bandpass Filtering of Neural data to create Discrete Frequency Band Features

    Strategy:
    ---------
        The Following code Band Pass Filters Discrete frequency bands of the neural data and outputs a
    List of each Channel with each element corresponding to a np.array of Time(row) vs. Frequencies(column)

    Parameters:
    -----------
    Channels: list [Song Length (Samples) x Channel #]
        Input Neural Data
    SN_L: int
        Stereotyped Length (In Samples) of Motif
    Gp_L: int
        Length of Time Buffer Before and After Motif
    StepSize: int (Optional)
        Bandwidth of Stepping (or Sliding) Bandpass filter, defaults to 20 Hz
    Lowest: int (Optional)
        Lowest Frequency Band for Bandpass Filter, defaults to 0 [In Other words The Frequency the Stepping Filter starts at]
    Order: int (Optional)
            Order of the Filter used, defaults to 175. [If FiltFilt = True then 350]
    fs: int (Optional)
        Sample Frequency of Neural Data, defaults to 1 KHz
    FiltFilt: bool (Optional)
        Controls whether to Filter Twice to Remove Phase Distortion, defaults to True
        [FiltFIlt performs zero-phase digital filtering by processing the input data, Channels, in both the forward and reverse directions.]
    Slide: bool (Optional)
        If True the Bandpass Filter will Slide by 1 Hz instead of the StepSize, defaults to False
    SUPP: bool (Optional)
        If True the Codes Status Prints will be suppressed, defaults to False

    Returns:
    --------
    Freq_Bins: list
        List of Resulting Bandpass Filtered Neural Data per channel
        [ch]->[Song Length (Samples) x Freq. Bin]
    Top: list
        List of High Frequency Cuttoffs of Bandpass's used
    Bottom: list
        List of Low Frequency Cutoffs
    '''

    Freq_Bins = []  # For holding the Bandpass Filtered Data
    Top, Bottom = Create_Bands(StepSize=StepSize, Lowest=Lowest, Slide=Slide, Suppress=SUPP)
    B = len(Top)  # Number of Frequency Bins
    Ch_Num = len(Channels[0, :])

    ## Band Pass and Isolate each Frequency Band
    for i in xrange(Ch_Num):
        Test = Channels[:, i]  # Grab Raw Signal of Select Channel
        Freq_Bins_Holder = np.zeros([SN_L + Gp_L, B])  # Initiate a Dynamic Sized Memory Space for Frequency Bins
        for l in xrange(0, B):
            if FiltFilt == True:
                Freq_Bins_Holder[:, l] = bandpass_filter(Test, Bottom[l], Top[l], fs, order_num=Order)
            if FiltFilt == False:
                Freq_Bins_Holder[:, l] = bandpass_filter_causal(Test, Bottom[l], Top[l], fs, order_num=Order)
        Freq_Bins.append(Freq_Bins_Holder[:, :])
    return Freq_Bins, Top, Bottom


# Pull out the Established frequency bands from Neural Data
## !!!!!!!!!! Add Functionality to alter these frequency bands !!!!!!!!!!!

def Generic_BPF(Channels, SN_L=int, Gp_L=int, Brain_waves=None, Order=175, fs=1000, FiltFilt=True):
    '''Bandpass Filter Neural data using Frequency Bands Described in literature (Pulled from Wikipedia)

    Strategy:
    ---------
        The Following code Bandpass Filters Established Frequency Bands of the neural data and outputs a
    List of each Channel with each element corresponding to a np array of Time(row) vs. Frequencies(column)

    Default Neural Oscillation Frequency bands from literature (WIkipedia):
       Brain waves
        - Delta wave – (0.2 – 3 Hz)
        - Theta wave – (4 – 7 Hz)
        - Alpha wave – (8 – 13 Hz)
        - Mu wave – (7.5 – 12.5 Hz)
        - SMR wave – (12.5 – 15.5 Hz)
        - Beta wave – (16 – 31 Hz)
        - Gamma wave – (32 – 100 Hz)

    #### Others: delta (1–4 Hz), theta (4–8 Hz), beta (13–30 Hz),low gamma (30–70 Hz), and high gamma (70–150 Hz)

    Parameters:
    -----------
    Channels: list [Song Length (Samples) x Channel #]
        Input Neural Data
    SN_L: int
        Stereotyped Length (In Samples) of Motif
    Gp_L: int
        Length of Time Buffer Before and After Motif
    Brain_waves: UNDER DEVELOPMENT (Idealy a class that can be pass into the function)
    Order: int (Optional)
            Order of the Filter used, defaults to 175. [If FiltFilt = True then 350]
    fs: int (Optional)
        Sample Frequency of Neural Data, defaults to 1 KHz
    FiltFilt: bool (Optional)
        Controls whether to Filter Twice to Remove Phase Distortion, defaults to True
        [FiltFIlt performs zero-phase digital filtering by processing the input data, Channels, in both the forward and reverse directions.]

    Returns:
    --------
    Freq_Bins: list [ch]->[Song Length (Samples) x Freq. Bin]
        List of Resulting Bandpass Filtered Neural Data per channel
    Top: list
        List of High Frequency Cuttoffs of Bandpass's used
    Bottom: list
        List of Low Frequency Cutoffs
    '''

    Freq_Bins = []  # For holding the Bandpass Filtered Data
    Top = [1, 4, 8, 13, 30, 70]
    Bottom = [4, 7, 13, 30, 70, 150]

    ## Ideal way to pass Brain_waves [UNDER DEVELOPMENT]
    # Top = Brain_waves.Top
    # Bottom = Brain_waves.Bottom

    B = len(Top)  # Number of Frequency Bins
    Ch_Num = len(Channels[0, :])

    ## Band Pass and Isolate each Frequency Band
    for i in xrange(Ch_Num):
        Test = Channels[:, i]  # Grab Raw Signal of Select Channel
        Freq_Bins_Holder = np.zeros([SN_L + Gp_L, B])  # Initiate a Dynamic Sized Memory Space for Frequency Bins
        for l in xrange(0, B):
            if FiltFilt == True:
                Freq_Bins_Holder[:, l] = bandpass_filter(Test, Bottom[l], Top[l], fs, order_num=Order)
            if FiltFilt == False:
                Freq_Bins_Holder[:, l] = bandpass_filter_causal(Test, Bottom[l], Top[l], fs, order_num=Order)
        Freq_Bins.append(Freq_Bins_Holder[:, :])
    return Freq_Bins, Top, Bottom


# Development for Re-Reference Data Command
# *********** This Originally Hardcoded the Exclusion of two Channels [NEED to Make Dyanmic in Future] ***********#
# **Added 5/23/2017**#
# Additional Functionality Added= Sliding Set Band Width [5/23/2017]

def RR_Neural(Frequencies, Good_Channels, Lowest=0, StepSize=20, SN_L=int, Gp_L=int, Slide=False):
    '''Re-reference All Frequencies on All Channels for ONE Behavioral Trial. *Must Be Run For Each Trial*

    Strategy:
    ---------
        Does this by concatenating each Channels concurrent activiy, taking the mean,
    then adding the mean to the channel Re-Reference template

    Steps:
    ------
        [1] Creates Template that contains the mean of each Frequency
        [2] Subtract Template from Each Channel

    Parameters:
    -----------
    Frequencies: list
        List of Channels (Each element is Frequency Bands vs. Time)
        [Ch]->[Frequency Bands x Time (Samples)]
    Good_Channels: list
        List of Channels to be Included in Re-Referencing
    Lowest: int (Optional)
        Lowest Frequency Band for Bandpass Filter, defaults to 0 [In Other words The Frequency the Stepping Filter starts at]
    StepSize: int (Optional)
        Bandwidth of Stepping (or Sliding) Bandpass filter, defaults to 20 Hz
    SN_L: int
        Stereotyped Length (In Samples) of Motif
    Gp_L: int
        Length of Time Buffer Before and After Motif
    Slide: bool (Optional)
        If True the Bandpass Filter will Slide by 1 Hz instead of the StepSize, defaults to False

    Returns:
    --------
    Freq_Bins_rr: list
        Re-Referenced Neural Data, Each element of the list is a np.array of Re-Referenced Neural Data
        [Ch]->[Time (Samples) x Frequency Band]
    Avg_Freq_Bins_LFP: np.array
        Array of the Mean Activity of each frequency band accross all included Channels
        [Time (Samples) x Frequency Band]
    '''

    ##[0] Initialize Variables for Filtering and Channel Exclusion

    if Slide == False:
        binwidths = np.arange(Lowest, 200, StepSize)  # Create Np Array of Bin Starts (Bottoms)

    if Slide == True:
        binwidths = np.arange(Lowest, 201 - StepSize, 1)  # Create Np Array of Bin Starts (Bottoms)

    B = len(binwidths)  # Determine the number of Freq. Bands

    Freq_Bins_rr = []  # For The RR'ed Band Passed Filtered Data

    ##[1] Find the Average for each Frequency Band over all Channels
    # 1.1 Prep
    Avg_Freq_Bins_LFP = np.zeros([SN_L + Gp_L, B])  # Initiate the Memory for the Mean Array

    # 1.2 Active Step
    for l in xrange(0, B):
        Ch_Freq_Bins_Holder = np.zeros(
            [SN_L + Gp_L, Z])  # Initiate Memory for Holding 1 Frequency Band for All Channels

        for i in xrange(len(Good_Channels)):  # Iterate over List of Good Channels
            Holder = Frequencies[Good_Channels[i]]  # Create Temporary Copy(Soft) of Frequencies
            Ch_Freq_Bins_Holder[:, i] = Holder[:, l]  # Store Specified Frequency of the Iterated Channel
        Avg_Freq_Bins_LFP[:, l] = Ch_Freq_Bins_Holder.mean(axis=1)  # Take Mean of Collective Frequencies Trace

    ##[2] Rereference using the Average Array
    for i in range(len(Frequencies)):  # Iterate over all Channels
        Freq_Bins_rr.append(
            Frequencies[i] - Avg_Freq_Bins_LFP[:, :])  # Subtract Mean of all Freq Bands from Iterated Channel

    return Freq_Bins_rr, Avg_Freq_Bins_LFP


# Created for Flexible Handling of Bad Channels for Re-Referencing
# Added on 5/23/2017

def Good_Channel_Index(Num_of_Channels, Bad_Channels):
    ''' Creates List of Good Channels

    Parameters:
    -----------
    Num_Channels: int
        Total Number of Channels
    Bad_Channels: list
        List of Bad Channels, channels to be excluded from Re-Referencing

    Returns:
    --------
    Good_Channels: list
        List of Channels to use for Re-Referencing
    '''
    Available_Channels = np.arange(Num_of_Channels)  # Create List of Recording Channels
    print 'All Channels:'
    print Available_Channels
    print 'All Good Channels:'
    Good_Channels = list(np.delete(Available_Channels, Bad_Channels))  # Remove Bad Channels from Total List of Channels
    print Good_Channels
    return Good_Channels


###### Last Working Here
####**** Look at Prep_pipeline to gain a better idea of how the Pipeline Currently Works and Areas for Improvement****
#### Also Note that the Current handling of Frequency Band Clippings is Bulky and May call for a more...
#### Efficient and Flexible Implementation

# Combinination of the Development for Normalization (Z-score) Command
# Updated for New Format 5/11/2017 [Changed: Z_Scored based on Song and Silence]

# [X]5/11/2017: Find a better way to index the Silence such that it is random

def Z_Score_data(Frequencies_Song, Frequencies_Silence, Numb_Motifs, Numb_Silence, Lowest=0, StepSize=20,
                 Slide=False):  # Eventually the Numb_Motifs needs to be changed
    ''' Z-Score Based on Neural Activity during Both Song and Silence

    Equation Used: z = (x – μ) / σ

    Description:
    ------------
    First Half of Code:
        [1] Initialize variables
        [2] Create Variable for Indexing Silence and Number of Bin
        [3] Line Up all of the Frequencies across All Trials for that Channel
            [3.1] Index into each Motif
            [3.2] Grab a selected Frequency Band
        [4] Find Mean
        [5] Find Std. Deviation
        [6] Store Both to a list
        Note: The Number of Silences and Songs do not need to be equal. An equal number of
        Silence will be randomly selected.

    Parameters:
    -----------
    Frequencies_Song: list
        Neural Activity during Song Trials
        [Trial]->[Ch]->[Frequency Bands x Time (Samples)]
    Frequencies_Silence: list
        Neural Activity during all Silence Trials
        [Trial]->[Ch]->[Frequency Bands x Time (Samples)]
    Numb_Motifs: int
        Number of Motifs in data set
    Numb_Silence: int
        Number of Examples of Silence
    Lowest: int (Optional)
        Lowest Frequency Band for Bandpass Filter, defaults to 0 [In Other words The Frequency the Stepping Filter starts at]
    StepSize: int (Optional)
        Bandwidth of Stepping (or Sliding) Bandpass filter, defaults to 20 Hz
    Slide: bool (Optional)
        If True the Bandpass Filter will Slide by 1 Hz instead of the StepSize, defaults to False

    Returns:
    --------
    Z_Scored_Data_Song: list
        Sample Based Z-Score of Neural Data during Song
        [Trials]->[Ch]->[Frequency Bands x Time (Samples)]
    Z_Scored_Data_Sqd_Song: list
        Sample Based Squared Z-Score of Neural Data during Song
        [Trials]->[Ch]->[Frequency Bands x Time (Samples)]
    Z_Scored_Data_Silence: list
        Sample Based Z-Score of Neural Data during Silence
        [Trials]->[Ch]->[Frequency Bands x Time (Samples)]
    Z_Scored_Data_Sqd_Silence: list
        Sample Based Squared Z-Score of Neural Data during Silence
        [Trials]->[Ch]->[Frequency Bands x Time (Samples)]
    Chan_Mean: list
        List of Each Channels Mean Over all Trials for Each Frequency Band
        [Ch]->[1 (Mean) x Freq. Band]
    Chan_StdDev: list
        List of Each Channels Standard Deviation Over all Trials for Each Frequency Band
        [Ch]->[1 (Mean) x Freq. Band]
    '''

    # [1] Initialize variables
    Freq_Bins_norm = []  # For Normalized Data
    Chan_Mean = []  # For Storing Means
    Chan_StdDev = []  # For Storing Standard Deviation

    # [2] Create Variable for Indexing Silence and Number of Bin
    Silence_Index = random.sample(xrange(Numb_Silence),
                                  Numb_Motifs)  # Create Index for Silence the same size as Numb_Motifs

    if Slide == False:
        binwidths = np.arange(Lowest, 200, StepSize)  # Create Np Array of Bin Starts (Bottoms)
    if Slide == True:
        binwidths = np.arange(Lowest, 201 - StepSize, 1)  # Create Np Array of Bin Starts (Bottoms)

    # [3] Line Up all of the Frequencies across All Trials for that Channel
    for i in range(len(Frequencies_Song[0])):  # Index over each Channel
        Chan_Holder = np.zeros((1, len(binwidths)))
        for k in xrange(Numb_Motifs):  # Index over each Motif Example
            Current_Chan_Song = Frequencies_Song[k][i]  # Grab Song Motifs
            Current_Chan_Silence = Frequencies_Silence[Silence_Index[k]][i]  # Grab Silence Examples
            Chan_Holder = np.concatenate((Chan_Holder, Current_Chan_Song), axis=0)  # Line Up all Song Trials
            Chan_Holder = np.concatenate((Chan_Holder, Current_Chan_Silence), axis=0)  # Line Up All Silence Trials

        # [4] & [5] Find Mean and Std. Deviation
        Chan_Holder = np.delete(Chan_Holder, 0, 0)  # Delete the First Row (Initialized Row)
        Chan_Means = np.mean(Chan_Holder, axis=0)  # Find Mean of Each Freq Band [Columns]
        Chan_StdDevs = np.std(Chan_Holder, axis=0)  # Find Std. of Each Freq Band [Columns]

        # [6] Store Both to List
        Chan_Mean.append(Chan_Means)
        Chan_StdDev.append(Chan_StdDevs)

    # Finds the Z-Score of all Neural Data
    #########################################################################
    # Actually Find the Z-score

    # Equation Used: z = (x – μ) / σ
    # [7] Initialize Variables
    Z_Scored_Data_Song = []  # For Z-scored Data
    Z_Scored_Data_Sqd_Song = []  # For sqd. Z-scored Data

    for k in xrange(Numb_Motifs):  # Index over Motifs
        Current_Motif = Frequencies_Song[k]  # Copy Current Trials data (Soft)
        z_scores = []  # Create Empty List
        z_scores_Sqd = []  # Create Empty List
        for i in range(len(Frequencies_Song[0])):  # Index over each Channel
            Current_Chan = Current_Motif[i]  # Copy Specific Channel for Given Trial
            z_scored = np.true_divide((Current_Chan - Chan_Mean[i]), Chan_StdDev[i])  # Calculate Z-Score
            z_scores.append(z_scored)  # Append Channel's Z-Score Value to List
            z_scores_Sqd.append(z_scored ** 2)  # Append Channel's Squared Z-Score Value to List
        Z_Scored_Data_Song.append(z_scores)  # Append Trial's Z-Score Value to List
        Z_Scored_Data_Sqd_Song.append(z_scores_Sqd)  # Append Trials's Squared Z-Score Value to List

    ## Repeat for Silence
    Z_Scored_Data_Silence = []  # For Z-scored Data
    Z_Scored_Data_Sqd_Silence = []  # For sqd. Z-scored Data
    for k in xrange(Numb_Motifs):  # Index over Motifs
        Current_Motif = Frequencies_Silence[k]  # Copy Current Trials data (Soft)
        z_scores = []  # Create Empty List
        z_scores_Sqd = []  # Create Empty List
        for i in range(len(Frequencies_Song[0])):  # Index over each Channel
            Current_Chan = Current_Motif[i]  # Copy Specific Channel for Given Trial
            z_scored = np.true_divide((Current_Chan - Chan_Mean[i]), Chan_StdDev[i])  # Calculate Z-Score
            z_scores.append(z_scored)  # Append Channel's Z-Score Value to List
            z_scores_Sqd.append(z_scored ** 2)  # Append Channel's Squared Z-Score Value to List
        Z_Scored_Data_Silence.append(z_scores)  # Append Trial's Z-Score Value to List
        Z_Scored_Data_Sqd_Silence.append(z_scores_Sqd)  # Append Trials's Squared Z-Score Value to List

    return Z_Scored_Data_Song, Z_Scored_Data_Sqd_Song, Z_Scored_Data_Silence, Z_Scored_Data_Sqd_Silence, Chan_Mean, Chan_StdDev


# ******* Skipped for now (1/30/2018)
# *** This Function will get deprecated when Python Updates
# Smoothing Function Accross Each Motifs

# Index into each Motif
# Smooth using pd.Series(x).rolling(window=2).mean()

def Smooth_data(Data, Numb_Motifs, Window=int):
    win_len = Window

    Smoothed_Data = []

    for k in xrange(Numb_Motifs):  # Index for Motifs
        Current_Motif = Data[k]
        sel_smoothed = []
        for i in range(16):  # Index for Channels
            #         z_scores= []
            Current_Chan = Current_Motif[i]
            #         for j in xrange(0, B_Need2Change): # Index for Freq Bands
            #             Current_FreqBand = Current_Chan(:,j)
            smoothed = pd.DataFrame(Current_Chan).rolling(window=win_len).mean().as_matrix()
            sel_smoothed.append(smoothed)

        Smoothed_Data.append(sel_smoothed)
    return Smoothed_Data


# Note: Only Describing Not Annotating to Prevent Future Pains [Use Class Function Moving Forward] 1/30/2018

# Entire Preprocessing Pipeline Command
# Updated for New Format 5/11/2017 [Changed: Z_Scored based on Song and Silence]
# Added Functionality to Handel Sliding of LFP Band (5/23/2017)

# ****** THIS SHOULDN'T BE BIASED BY Motif Labels

def Prep_pipeline(Song_Data, Numb_Motifs, Silence_Data, Numb_Silence, Good_Chan, SN_L=500, Gp_L=4000, StepSize=15,
                  Lowest=5, Order=175, fs=1000, window=10, filtfilt=True, Sliding=False, Suppress=False):
    '''

    Parameters:
    -----------
    Song_Data: list
        Neural Activity during Song Trials
        [Trial]->[Ch]->[Frequency Bands x Time (Samples)]
    Numb_Motifs: int
        Number of Motifs in data set
    Silence_Data: list
        Neural Activity during all Silence Trials
        [Trial]->[Ch]->[Frequency Bands x Time (Samples)]
    Numb_Silence: int
        Number of Examples of Silence
    Good_Chan: list
        List of Channels to use for Re-Referencing
    SN_L: int (Optional)
        Stereotyped Length (In Samples) of Motif, defaults to 500
    Gp_L: int (Optional)
        Length of Time Buffer Before and After Motif, defaults to 4000
    StepSize: int (Optional)
        Bandwidth of Stepping (or Sliding) Bandpass filter, defaults to 15 Hz
    Lowest: int (Optional)
        Lowest Frequency Band for Bandpass Filter, defaults to 5 [In Other words The Frequency the Stepping Filter starts at]
    Order: int (Optional)
        Order of the Filter used, defaults to 175. [If FiltFilt = True then 350]
    fs: int (Optional)
        Sample Frequency of Neural Data, defaults to 1000 (1 KHz)
    window: int (Optional)
        , defaults to 10
    filtfilt: bool (Optional)
        Controls whether to Filter Twice to Remove Phase Distortion, defaults to True
        [FiltFIlt performs zero-phase digital filtering by processing the input data, Channels, in both the forward and reverse directions.]
    Sliding: bool (Optional)
        If True the Bandpass Filter will Slide by 1 Hz instead of the StepSize, defaults to False
    Suppress: bool (Optional)
        If True the Codes Status Prints will be suppressed, defaults to False

    Returns:
    --------
    All_Steps_Song: List
        List of Each Step of Pre-Processing done on Song Neural Data
        [BPF                 ]->[Trials]->[Ch]->[Time (Samples) x Freq. Bands]
        [BPF_RR              ]
        [Z-Scored            ]
        [z-Scored Squared    ]
        [Smothed Z-Scored    ]
        [Smothed z-Scored Sq.]
    All_Steps_Silence: List
        List of Each Step of Pre-Processing done on Silence Neural Data
        [BPF                 ]->[Trials]->[Ch]->[Time (Samples) x Freq. Bands]
        [BPF_RR              ]
        [Z-Scored            ]
        [z-Scored Squared    ]
        [Smothed Z-Scored    ]
        [Smothed z-Scored Sq.]
    '''
    # Need to Improve SN_L & Gp_L handling

    BPF_Motifs = []
    BPF_RR_Motifs = []

    for i in xrange(Numb_Motifs):  # [x]Eventually Change Numb_Motifs to a Flexible Index of Good_Motifs

        # First: Sliding Band Pass Filters of All Motifs
        Sel_Freq, Top, Bottom = Sliding_BPF(Song_Data[i], SN_L=SN_L,
                                            Gp_L=Gp_L, StepSize=StepSize, Lowest=Lowest, Order=Order, fs=fs,
                                            FiltFilt=filtfilt, Slide=Sliding, SUPP=Suppress)
        BPF_Motifs.append(Sel_Freq)

        # Second: Re-Reference Each Motif
        RR_Test_Freq, Avg_Test_Freq = RR_Neural(Sel_Freq, Good_Channels=Good_Chan, Lowest=Lowest, StepSize=StepSize,
                                                SN_L=SN_L, Gp_L=Gp_L, Slide=Sliding)
        BPF_RR_Motifs.append(RR_Test_Freq)

    ########## [Updated for New Format 5/11/2017] ######################
    # Run Again for Silence
    BPF_Silence = []
    BPF_RR_Silence = []

    for i in xrange(Numb_Silence):  # [x]Eventually Change Numb_Motifs to a Flexible Index of Good_Motifs

        # First: Sliding Band Pass Filters of All Motifs
        Sel_Freq, Top, Bottom = Sliding_BPF(Silence_Data[i], SN_L=SN_L,
                                            Gp_L=Gp_L, StepSize=StepSize, Lowest=Lowest, Order=Order, fs=fs,
                                            FiltFilt=filtfilt, Slide=Sliding, SUPP=Suppress)
        BPF_Silence.append(Sel_Freq)

        # Second: Re-Reference Each Motif
        RR_Test_Freq, Avg_Test_Freq = RR_Neural(Sel_Freq, Good_Channels=Good_Chan, Lowest=Lowest, StepSize=StepSize,
                                                SN_L=SN_L, Gp_L=Gp_L, Slide=Sliding)
        BPF_RR_Silence.append(RR_Test_Freq)

    ##################################################################
    # Third: Normalize
    Z_Scored_Data_Song, Z_Scored_Data_Sqd_Song, Z_Scored_Data_Silence, Z_Scored_Data_Sqd_Silence, Chan_Mean, Chan_StdDev = Z_Score_data(
        BPF_RR_Motifs, BPF_RR_Silence,
        Numb_Motifs, Numb_Silence,
        Lowest=Lowest,
        StepSize=StepSize,
        Slide=Sliding)

    # Fourth: Smooth
    Sm_Z_Scored_Data_Song = Smooth_data(Z_Scored_Data_Song, Numb_Motifs, Window=window)
    Sm_Z_Scored_Data_Sqd_Song = Smooth_data(Z_Scored_Data_Sqd_Song, Numb_Motifs, Window=window)

    ##Repeat for Silence
    Sm_Z_Scored_Data_Silence = Smooth_data(Z_Scored_Data_Silence, Numb_Motifs, Window=window)
    Sm_Z_Scored_Data_Sqd_Silence = Smooth_data(Z_Scored_Data_Sqd_Silence, Numb_Motifs, Window=window)

    # Create a single list that has all of the steps of the pipeline stored
    #     return BPF_Motifs, BPF_RR_Motifs, Z_Scored_Data, Z_Scored_Data_Sqd, Sm_Z_Scored_Data, Sm_Z_Scored_Data_Sqd

    All_Steps_Song = []
    All_Steps_Song.append(BPF_Motifs)
    All_Steps_Song.append(BPF_RR_Motifs)
    All_Steps_Song.append(Z_Scored_Data_Song)
    All_Steps_Song.append(Z_Scored_Data_Sqd_Song)
    All_Steps_Song.append(Sm_Z_Scored_Data_Song)
    All_Steps_Song.append(Sm_Z_Scored_Data_Sqd_Song)

    ## Repeat for Silence
    All_Steps_Silence = []
    All_Steps_Silence.append(BPF_Silence)
    All_Steps_Silence.append(BPF_RR_Silence)
    All_Steps_Silence.append(Z_Scored_Data_Silence)
    All_Steps_Silence.append(Z_Scored_Data_Sqd_Silence)
    All_Steps_Silence.append(Sm_Z_Scored_Data_Silence)
    All_Steps_Silence.append(Sm_Z_Scored_Data_Sqd_Silence)

    return All_Steps_Song, All_Steps_Silence

