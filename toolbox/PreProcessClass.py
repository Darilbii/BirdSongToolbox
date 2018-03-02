#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from functools import wraps

import numpy as np

from PreProcTools import bandpass_filter, bandpass_filter_causal, Create_Bands, Good_Channel_Index


# Master Function: Handles Flexible Bandpass Filtering

def BPF_Module(Channels, Freq_Bands=tuple, SN_L=int, Gp_L=int, Num_Chan=int, Num_Freq=int, order_num=175, fs=1000,
               FiltFilt=True):
    '''Bandpass Filter Neural data using User Defined Frequency Bands for ONE Trials

    Strategy:
    ---------
        The Following code Bandpass Filters User Defined Frequency Bands of the neural data and outputs a
    List of each Channel with each element corresponding to a np array of Time(row) vs. Frequencies(column)

    Parameters:
    -----------
    Channels: list [Song Length (Samples) x Channel #]
        Input Neural Data
    Freq_Bands: tuple
        Cuttoff Frequencies of Passband for Band Pass Filters, Components of Tuple are Tops and Bottoms which are lists
        of the High Frequency and Low Frequency Cutoffs, respectively, of the Pass Bands
        ([Tops], [Bottoms])
    SN_L: int
        Stereotyped Length (In Samples) of Motif
    Gp_L: int
        Length of Time Buffer Before and After Motif
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
    '''
    Top, Bottom = Freq_Bands  # Create Variable for Pass Band Boundaries
    Freq_Bins = []  # For holding the Bandpass Filtered Data

    ## Band Pass and Isolate each Frequency Band
    for i in xrange(Num_Chan):
        Test = Channels[:, i]  # Grab Raw Signal of Select Channel
        Freq_Bins_Holder = np.zeros([SN_L + Gp_L, Num_Freq])  # Initiate a Dynamic Sized Memory Space for Frequency Bins
        for l in xrange(0, Num_Freq):
            if FiltFilt == True:
                Freq_Bins_Holder[:, l] = bandpass_filter(Test, Bottom[l], Top[l], fs, order_num=order_num)
            if FiltFilt == False:
                Freq_Bins_Holder[:, l] = bandpass_filter_causal(Test, Bottom[l], Top[l], fs, order_num=order_num)
        Freq_Bins.append(Freq_Bins_Holder[:, :])
    return Freq_Bins


def BPF_Master(Channels, Num_Trials, Freq_Bands=tuple, SN_L=int, Gp_L=int, Num_Chan=int, Num_Freq=int, order_num=175,
               fs=1000, FiltFilt=True):
    '''Bandpass Filter Neural data using User Defined Frequency Bands for All Trials

    Strategy:
    ---------
        The Following code Bandpass Filters User Defined Frequency Bands of the neural data and outputs a
    List of each Channel with each element corresponding to a np array of Time(row) vs. Frequencies(column)

    Parameters:
    -----------
    Channels: list [Song Length (Samples) x Channel #]
        Input Neural Data
    Num_Trials: int
        Number of Trials for Behavior
    Freq_Bands: tuple
        Cuttoff Frequencies of Passband for Band Pass Filters, Components of Tuple are Tops and Bottoms which are lists
        of the High Frequency and Low Frequency Cutoffs, respectively, of the Pass Bands
        ([Tops], [Bottoms])
    SN_L: int
        Stereotyped Length (In Samples) of Motif
    Gp_L: int
        Length of Time Buffer Before and After Motif
    Num_Chan: int
        Number of Recording Channels
    Order: int (Optional)
            Order of the Filter used, defaults to 175. [If FiltFilt = True then 350]
    fs: int (Optional)
        Sample Frequency of Neural Data, defaults to 1 KHz
    FiltFilt: bool (Optional)
        Controls whether to Filter Twice to Remove Phase Distortion, defaults to True
        [FiltFIlt performs zero-phase digital filtering by processing the input data, Channels, in both the forward and reverse directions.]

    Returns:
    --------
    BPF_Motifs: list
        List of All Trial's Resulting Bandpass Filtered Neural Data per channel
        [Trial]->[ch]->[Song Length (Samples) x Freq. Bin]
    '''
    BPF_Motifs = []
    for i in xrange(Num_Trials):
        BPF_Motifs.append(
            BPF_Module(Channels[i], Freq_Bands=Freq_Bands, SN_L=SN_L, Gp_L=Gp_L, Num_Chan=Num_Chan, Num_Freq=Num_Freq,
                       order_num=order_num, fs=fs, FiltFilt=FiltFilt))
    return BPF_Motifs


def RR_Neural_Module(Frequencies, Good_Channels, Num_Freq, SN_L=int, Gp_L=int):
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
        Single Trial's Dataset Consisting of a List of Channels (Each element is Frequency Bands vs. Time)
        [Ch]->[Frequency Bands x Time (Samples)]
    Good_Channels: list
        List of Channels to be Included in Re-Referencing
    Num_Freq: int
        Number of Frequency Bands in Dataset
    SN_L: int
        Stereotyped Length (In Samples) of Motif
    Gp_L: int
        Length of Time Buffer Before and After Motif
    Returns:
    --------
    Freq_Bins_rr: list
        Re-Referenced Neural Data, Each element of the list is a np.array of Re-Referenced Neural Data
        [Ch]->[Time (Samples) x Frequency Band]
    Avg_Freq_Bins_LFP: np.array
        Array of the Mean Activity of each frequency band accross all included Channels
        [Time (Samples) x Frequency Band]
    '''
    Freq_Bins_rr = []  # For The RR'ed Band Passed Filtered Data

    ##[1] Find the Average for each Frequency Band over all Channels
    # 1.1 Prep
    Avg_Freq_Bins_LFP = np.zeros([SN_L + Gp_L, Num_Freq])  # Initiate the Memory for the Mean Array

    # 1.2 Active Step
    for l in xrange(0, Num_Freq):
        Ch_Freq_Bins_Holder = np.zeros(
            [SN_L + Gp_L, len(Good_Channels)])  # Initiate Memory for Holding 1 Frequency Band for All Good Channels

        for i in xrange(len(Good_Channels)):  # Iterate over List of Good Channels
            Holder = Frequencies[Good_Channels[i]]  # Create Temporary Copy(Soft) of Frequencies
            Ch_Freq_Bins_Holder[:, i] = Holder[:, l]  # Store Specified Frequency of the Iterated Channel
        Avg_Freq_Bins_LFP[:, l] = Ch_Freq_Bins_Holder.mean(axis=1)  # Take Mean of Collective Frequencies Trace

    ##[2] Rereference using the Average Array
    for i in range(len(Frequencies)):  # Iterate over all Channels
        Freq_Bins_rr.append(
            Frequencies[i] - Avg_Freq_Bins_LFP[:, :])  # Subtract Mean of all Freq Bands from Iterated Channel

    return Freq_Bins_rr, Avg_Freq_Bins_LFP


def RR_Neural_Master(Frequencies, Num_Trials, Good_Channels, Num_Freq, SN_L=int, Gp_L=int):
    '''Re-reference All Frequencies on All Channels for All Behavioral Trials.

    Strategy:
    ---------
        Iteratively Runs RR_Neural_Module and appends the results

    Parameters:
    -----------
    Frequencies: list
        List of All Trial's Dataset Consisting of a List of Channels (Each element is Frequency Bands vs. Time)
        [Trials]->[Ch]->[Frequency Bands x Time (Samples)]
    Num_Trials: int
        Number of Trials of the Input Data's Behavior
    Good_Channels: list
        List of Channels to be Included in Re-Referencing
    Num_Freq: int
        Number of Frequency Bands in Dataset
    SN_L: int
        Stereotyped Length (In Samples) of Motif
    Gp_L: int
        Length of Time Buffer Before and After Motif
    Returns:
    --------
    Freq_Bins_rr: list
        Re-Referenced Neural Data, Each element of the list is a np.array of Re-Referenced Neural Data
        [Trials]->[Ch]->[Time (Samples) x Frequency Band]
    Avg_Freq_Bins_LFP: np.array
        Array of the Mean Activity of each frequency band accross all included Channels
        [Trials]->[Time (Samples) x Frequency Band]
    '''
    RR_Trials = []
    Avg_Freq_RR_Trials = []
    for i in xrange(Num_Trials):
        RR_Trial_hold, Avg_Freq_RR_Trial_hold = RR_Neural_Module(Frequencies[i], Good_Channels, Num_Freq, SN_L=SN_L,
                                                                 Gp_L=Gp_L)
        RR_Trials.append(RR_Trial_hold)
        Avg_Freq_RR_Trials.append(Avg_Freq_RR_Trial_hold)
    return RR_Trials, Avg_Freq_RR_Trials


def Find_Z_Score_Metrics(Frequencies_Song, Frequencies_Silence, Num_Freq, Numb_Motifs, Numb_Silence):
    ''' Find the Mean and Standard Deviation of Day's Recordings

    Description:
    ------------
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
    Num_Freq: int
        Number of Frequency Bands
    Numb_Motifs: int
        Number of Motifs in data set
    Numb_Silence: int
        Number of Examples of Silence
    Lowest: int (Optional)
        Lowest Frequency Band for Bandpass Filter, defaults to 0 [In Other words The Frequency the Stepping Filter starts at]

    Returns:
    --------
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
                                  Numb_Motifs)  # Create Index for Silence with same size as Song Trials

    # [3] Line Up all of the Frequencies across All Trials for that Channel
    for i in range(len(Frequencies_Song[0])):  # Index over each Channel
        Chan_Holder = np.zeros((1, Num_Freq))
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
    return Chan_Mean, Chan_StdDev


def Z_Score_Module(Frequencies, Num_Trials, Chan_Mean, Chan_StdDev):
    ''' Z-Score All of the Input Data using the Inputed Mean and Standard Deviation of All Channels for All Frequency Bands

    Equation Used:
    --------------
        z = (x – μ) / σ

    Parameters:
    -----------
    Frequencies: list
        Input Neural Activity during all Trials
        [Trial]->[Ch]->[Frequency Bands x Time (Samples)]
    Num_Trials: int
        Number of All Behavior Trials for Input Data
    Chan_Mean: list
        List of Each Channels Mean Over all Trials for Each Frequency Band
        [Ch]->[1 (Mean) x Freq. Band]
    Chan_StdDev: list
        List of Each Channels Standard Deviation Over all Trials for Each Frequency Band
        [Ch]->[1 (Mean) x Freq. Band]

    Returns:
    --------
    Z_Scored_Data: list
        Sample Based Z-Score of Input Neural Data
        [Trials]->[Ch]->[Frequency Bands x Time (Samples)]
    '''

    Z_Scored_Data = []  # For Z-scored Data

    # [2] Sample Based Z-Score
    for k in xrange(Num_Trials):  # Index over Motifs
        Current_Trial = Frequencies[k]  # Copy Current Trials data (Soft)
        z_scores = []  # Create Empty List
        for i in range(len(Frequencies[0])):  # Index over each Channel
            Current_Chan = Current_Trial[i]  # Copy Specific Channel for Given Trial
            z_scored = np.true_divide((Current_Chan - Chan_Mean[i]), Chan_StdDev[i])  # Calculate Z-Score
            z_scores.append(z_scored)  # Append Channel's Z-Score Value to List
        Z_Scored_Data.append(z_scores)  # Append Trial's Z-Score Value to List
    return Z_Scored_Data


def Z_Score_data_Master(Frequencies_Song, Frequencies_Silence, Numb_Freq, Numb_Motifs, Numb_Silence):
    ''' Z-Score Based on Neural Activity during Both Song and Silence

    Equation Used:
    --------------
        z = (x – μ) / σ

    Notes:
    ------
        The Number of Silences and Songs do not need to be equal. An equal number of Silence will be
        randomly selected for Z-Scoring, but the the Trial count for Silence will Remain the same

    Parameters:
    -----------
    Frequencies_Song: list
        Neural Activity during Song Trials
        [Trial]->[Ch]->[Frequency Bands x Time (Samples)]
    Frequencies_Silence: list
        Neural Activity during all Silence Trials
        [Trial]->[Ch]->[Frequency Bands x Time (Samples)]
    Numb_Freq: int
        Number of Frequency Bands
    Numb_Motifs: int
        Number of Motifs in data set
    Numb_Silence: int
        Number of Examples of Silence

    Returns:
    --------
    Z_Scored_Data_Song: list
        Sample Based Z-Score of Neural Data during Song
        [Song Trials]->[Ch]->[Frequency Bands x Time (Samples)]
    Z_Scored_Data_Silence: list
        Sample Based Z-Score of Neural Data during Silence
        [Silence Trials]->[Ch]->[Frequency Bands x Time (Samples)]
    Means: list
        List of Each Channels Mean Over all Trials for Each Frequency Band
        [Ch]->[1 (Mean) x Freq. Band]
    StdDevs: list
        List of Each Channels Standard Deviation Over all Trials for Each Frequency Band
        [Ch]->[1 (Mean) x Freq. Band]
    '''
    # Find Mean & Standard Deviation of All Frequencies on Each Channel
    Means, StdDevs = Find_Z_Score_Metrics(Frequencies_Song, Frequencies_Silence, Num_Freq=Numb_Freq,
                                          Numb_Motifs=Numb_Motifs, Numb_Silence=Numb_Silence)
    # Z-Score Song Trials
    Z_Scored_Data_Song = Z_Score_Module(Frequencies_Song, Num_Trials=Numb_Motifs, Chan_Mean=Means,
                                        Chan_StdDev=StdDevs)
    # Z-Score Silence Trials
    Z_Scored_Data_Silence = Z_Score_Module(Frequencies_Silence, Num_Trials=Numb_Silence, Chan_Mean=Means,
                                           Chan_StdDev=StdDevs)
    return Z_Scored_Data_Song, Z_Scored_Data_Silence, Means, StdDevs


# Pre-Processing Class Function 1/30/2018


### LAST WORKING HERE.... KINDA LOST AND BEEN WORKING DISTRACTEDLY ON
### VARIOUS THINGS MAY BE BEST TO START PASSING THIS ONTO GITHUB

import copy


class Pipeline():
    ''' Class for Pre-Processing Neural Data

    Description:
    ------------
    The Processing Functions all follow the same general Steps:
        - [1] Validate proper steps have been made and Necessary Object Instances exist
            - [1.1] Check Pipeline is still Open
            - [1.2] Check Dependencies Exist
        - [2] Back-up Neural Data in case of Mistake [Make_Backup(self)]
        - [3] Do User Specified Processing on Song Neural Data
        - [4] Do User Specified Processing on Silence Neural Data
        - [5] Update the Process Log with User Defined Steps (Done Last Incase of Error)

    Methods:
    --------
        Convenience:
        ------------
        .identity():
            Desplay Bird ID and Recording Date
        .Pipe_Steps():
            Desplay Pre-Processing Steps and Relevant Parameters
        .Restore():
            Undo Last Pre-Processing Step and Restore from Back-up

        Functional:
        -----------
        .Pipe_end():
            Close Pipeline and Prevent Accidental Editing of Data
        .Pipe_Reopen():
            Re-Open Pipeline for Further Pre-Processing

        Processing:
        -----------
        .Define_Frequencies(*Param):
            Define Method for Band Passing
        .Band_Pass_Filter():
            Band Pass Filter Data
        .Re_Reference():
            Re-Reference using a Common Average Reference Filter
        .Z_Score():
            Z-Score Input Data

    Objects:
    --------
    .bird_id: str
        Bird Indentifier to Locate Specified Bird's data folder
    .date: str
        Experiment Day to Locate it's Folder
    .Sn_Len = int
        Time Duration of Birds Motif (in Samples)
    .Gap_Len = int
        Duration of Buffer used for Trials (in Samples)
    .Num_Chan = int
        Number of Recording Channels used on Bird
    .Bad_Channels = list
        List of Channels with Noise to be excluded from Common Average Referencing
    .Fs = int
        Sample Frequency of Data (in Samples)
    .Song_Neural: list
        Lowpass Filtered Neural data during Song Trials (300 Hz. Cutoff)
        [Number of Trials]-> [Trial Length (Samples @ 1KHz) x Ch]
    .Song_Audio: list
        Audio of Trials, centered on motif
        [Number of Trials]-> [Trial Length (Samples @ 30KHz) x 1]
    .Silence_Neural: list
        Lowpass Filtered Neural data during Silent Trials (300 Hz. Cutoff)
        [Number of Trials]-> [Trial Length (Samples @ 1KHz) x Ch]
    .Silence_Audio: list
        Audio of Silents Trials
        [Number of Trials]-> [Trial Length (Samples @ 30KHz) x 1]
    .Num_Motifs: int
        Number of Motifs in data set
    .Num_Silence: int
        Number of Examples of Silence
    .Good_Motifs: list
        Index of All Good Motifs, 'Good' is defined as having little noise and no dropped (or missing) syllables
    .First_Motifs: list
        Index of All Good First Motifs, this motif is the first motif in a bout and is classified as 'Good'
    .Last_Motifs: list
        Index of All Good Last Motifs, this motif is the last motif in a bout and is classified as 'Good'
    .Bad_Motifs: list
        Index of All Bad Motifs with no dropped syllables, These motifs have interferring audio noise
    .LS_Drop: list
        Index of All Bad Motifs with the last syllable dropped, These motifs are classified as Bad
    .All_First_Motifs: list
        Index of All First Motifs in a Bout Regardless of Quality label, This is Useful for Clip-wise (Series) Analysis
    .Good_Channels: list
        List of Channels that are to be included in a Common Average Filter
    .All_Last_Motifs: list
        Index of All Last Motifs in a Bout Regardless of Quality label, This is Useful for Clip-wise (Series) Analysis
    .Good_Mid_Motifs: list
        Index of All Good Motifs in the middle of a Bout Regardless of Quality label, This is Useful for Clip-wise (Series)
        Analysis
    '''

    def __init__(self, Imported_Data):
        '''Initiallizes by hardcopying the input data for Pre-Processing'''
        # [1] Transfer over Data's Instances using Hard Copies
        self.bird_id = copy.deepcopy(Imported_Data.bird_id)
        self.date = copy.deepcopy(Imported_Data.date)
        self.Sn_Len = copy.deepcopy(Imported_Data.Sn_Len)
        self.Gap_Len = copy.deepcopy(Imported_Data.Gap_Len)
        self.Num_Chan = copy.deepcopy(Imported_Data.Num_Chan)
        self.Bad_Channels = copy.deepcopy(Imported_Data.Bad_Channels)  # Debating Hard Passing Bad_Channels
        self.Fs = copy.deepcopy(Imported_Data.Fs)
        self.Song_Audio = copy.deepcopy(Imported_Data.Song_Audio)  # Debating Including Audio
        self.Song_Neural = copy.deepcopy(Imported_Data.LPF_DS_Song)
        self.Silence_Audio = copy.deepcopy(Imported_Data.Silence_Audio)  # Debating Including Audio
        self.Silence_Neural = copy.deepcopy(Imported_Data.LPF_DS_Silence)
        self.Num_Motifs = copy.deepcopy(Imported_Data.Num_Motifs)
        self.Num_Silence = copy.deepcopy(Imported_Data.Num_Silence)
        self.Good_Motifs = copy.deepcopy(Imported_Data.Good_Motifs)
        self.Bad_Motifs = copy.deepcopy(Imported_Data.Bad_Motifs)
        self.LS_Drop = copy.deepcopy(Imported_Data.LS_Drop)
        self.Last_Motifs = copy.deepcopy(Imported_Data.Last_Motifs)
        self.First_Motifs = copy.deepcopy(Imported_Data.First_Motifs)
        self.All_First_Motifs = copy.deepcopy(Imported_Data.All_First_Motifs)
        self.All_Last_Motifs = copy.deepcopy(Imported_Data.All_Last_Motifs)
        self.Good_Mid_Motifs = copy.deepcopy(Imported_Data.Good_Mid_Motifs)
        self.Good_Channels = Good_Channel_Index(self.Num_Chan, self.Bad_Channels)
        

        # Create Processing Operator Instances
        self.Activity_Log = {}  # Initiate a Log of Activity for Recounting Processing Steps
        self.Backup = ()  # Back-up Neural Data in case Processing Error
        self.Status = True  # Value to Indicate whether Processing is Active
        self.Step_Count = 0  # Initiate Step Counter

    ########## Last Here
    def _StandardStep(func):
        '''Wraper for Processing Steps'''

        @wraps(func) # TODO: Make it Able to Return it's Description in the Help Magic view
        def steps(self, *args, **kwargs):
            assert self.Status == True, 'Pipe is Closed. To re-open use Pipe_Reopen'
            print 'Wrapper Worked' #TODO Edit this Decorator to Print Useful Strings
            self.Make_Backup()  # Back-up Neural Data in case of Mistake
            func(self, *args, **kwargs)  # Pre-Processing Function
            self.Update_Log(self.Log_String)  # Update Log
            del self.Log_String
            print 'Ooops I meant Decorator'

        return steps

    def Make_Backup(self):
        '''Quickly Backs Up Neural Data '''
        assert self.Status == True  # Evaluate Status of Data in Pipeline
        self.Backup = self.Song_Neural, self.Silence_Neural

    def Update_Log(self, step):
        '''Updates Log recording Processing Steps Implemented'''
        assert type(step) == str
        self.Step_Count = self.Step_Count + 1
        self.Activity_Log[self.Step_Count] = step

    def identity(self):
        '''Convenience Function: Displays the Bird ID and Recording Date'''
        print 'bird id: ' + self.bird_id
        print 'recording: ' + self.date

    def Pipe_Steps(self):
        '''Convenience Function: Prints Pipeline Steps Used'''
        assert len(self.Activity_Log) > 0, 'No Steps Implemented'
        for i in xrange(len(self.Activity_Log)):
            print str(i + 1) + ': ' + self.Activity_Log[i + 1]

    # noinspection PyTupleAssignmentBalance
    def Restore(self):
        '''Conveniece Function: Restores Neural Data to the Immediate Prior Step'''
        assert self.Step_Count > 0
        assert self.Status == True, 'Pipe Closed'  # Evaluate Status of Data in Pipeline
        assert type(self.Backup) == tuple
        self.Song_Neural, self.Silence_Neural = self.Backup
        self.Step_Count = self.Step_Count - 1  # Backtrack Step Count
        self.Activity_Log[
            self.Step_Count] = 'Restored to Previous'  # Replace Log with Holding Statement (Prevent Error)

    #     def Validate(self):
    #         '''Long step that makes sure Backup is sucessful'''
    #         assert (self.Song_Neural, self.Silence_Neural) == self.Backup

    def Pipe_end(self):
        '''Marks end of Pipeline. Prevents accidental steps after all Processing Steps are Implemented'''
        assert self.Status == True, 'Pipe Already Closed'
        print 'Pipeline Complete'
        self.Status = False
        del self.Backup

    def Pipe_Reopen(self):
        '''Re-Opens Pipeline for Further Processing'''
        assert self.Status == False, 'Pipe Already Open'
        print 'Pipeline Re-opened'
        self.Status = True
        self.Backup = ()

    # TODO: Make Sure you can't Restore Consectively
    # TODO: The Entire Restore Pipeline is Faulty
    # TODO: Change Discription of Gap_Len to: Total Length (Duration) of time Buffer around Trials (To Determine Buffer Before or After Divide by 2)
    # TODO: Top and Bottom annotation must be updated to ndarray (Benefit is they are immutable)


    def Define_Frequencies(self, Instructions, StepSize=20, Lowest=0, Slide=False, suppress=False):
        '''Creates Index for Frequency Pass Band Boundaries (High and Low Cuttoff Frequencies)

        Parameters:
        -----------
        Instructions: str or tuple
            Instructions on how to Bandpass Filter Neural Data, options are {tuple, 'Stereotyped' or 'Sliding'}
             - tuple: Custom Frequency Bands must be structured as ([Bottoms],[Tops])
             - 'Stereotyped': Frequency Bands Previously defined in literature (From Wikipedia)
             - 'Sliding': Sliding Band pass Filter that are further described by Optional Variables
        StepSize: int (Optional)
            Required if Instructions set to 'Sliding'
            Width of All Bandpass Filters (defaults to 20 Hz)
        Lowest: int (Optional)
            Required if Instructions set to 'Sliding'
            Lowest frequency to start (defaults to 0)
        Slide: bool (Optional)
            Required if Instructions set to 'Sliding'
            If True Bandpass Filters will have a stepsize of 1 Hz (Defaults to False)
        Suppress: bool (Optional)
            Required if Instructions set to 'Sliding'
            If True Function's print statements will be ignored (Defaults to False) [Helps to reduce unnecesary printing steps]

        Returns:
        --------
        .Top: list
            List of High Frequency Cuttoffs
        .Bottom: list
            List of Low Frequency Cutoffs
        '''
        assert self.Status == True, 'Pipe is Closed. This Function SHOULD NOT be run on its own'
        assert type(Instructions) == str or type(Instructions) == tuple  # Ensure Instructions follow Assumptions
        if type(Instructions) == tuple:
            Bottom, Top = Instructions  # Break Tuple into Top and Bottom
            assert type(Top) == list
            assert type(Bottom) == list
        if type(Instructions) == str:
            if Instructions == 'Stereotyped':
                Top = [4, 7, 13, 30, 70, 150]
                Bottom = [1, 4, 8, 13, 30, 70]
            if Instructions == 'Sliding':
                Top, Bottom = Create_Bands(StepSize=StepSize, Lowest=Lowest, Slide=Slide, Suppress=suppress)
        # Store Frequency Band Boundaries
        assert len(Top) == len(Bottom)  # Make Sure No Mismatch Errors
        self.Top = Top
        self.Bottom = Bottom
        self.Num_Freq = len(Top)

    @_StandardStep
    def Band_Pass_Filter(self, order_num=175, FiltFilt=True):
        ''' Bandpass Filter Data using User Defined Frequency Bands'''
        try:
            self.Top
        except NameError:
            print 'You Need to Define your Frequency Bands'
            print 'Try using .Define_Frequencies()'
        else:
            assert len(np.shape(
                self.Song_Neural)) == 3, 'You have Already Bandpass Filtered '  # BPF Changes Architecture and Cannot be run repeatedly in series.  It Should be Run First
            self.Song_Neural = BPF_Master(self.Song_Neural, Num_Trials=self.Num_Motifs,
                                          Freq_Bands=(self.Top, self.Bottom), SN_L=self.Sn_Len, Gp_L=self.Gap_Len,
                                          Num_Chan=self.Num_Chan, Num_Freq=self.Num_Freq, order_num=order_num,
                                          fs=self.Fs, FiltFilt=FiltFilt)
            self.Silence_Neural = BPF_Master(self.Silence_Neural, Num_Trials=self.Num_Silence,
                                             Freq_Bands=(self.Top, self.Bottom), SN_L=self.Sn_Len, Gp_L=self.Gap_Len,
                                             Num_Chan=self.Num_Chan, Num_Freq=self.Num_Freq, order_num=order_num,
                                             fs=self.Fs,
                                             FiltFilt=FiltFilt)

            # Construct Log Update Components
            if FiltFilt == True:
                Zero_Phase = 'with Zero Phase Distortion'
                Order = str(order_num * 2)
            else:
                Zero_Phase = 'with Phase Distortion (Causal)'
                Order = str(order_num)
            self.Log_String = 'Bandpass Filtered with ' + str(
                self.Num_Freq) + ' Filters ' + Zero_Phase + ' of Order: ' + Order  # Construct Log String (Combine Components)

    @_StandardStep
    def Re_Reference(self):
        '''Re-Reference Data using a Common Average Reference Filter that Excludes Channels Directed by User
        '''
        assert type(self.Good_Channels) == list, 'Something is Wrong with .Good_Channels'
        self.Song_Neural, self.Song_CAR = RR_Neural_Master(self.Song_Neural, Num_Trials=self.Num_Motifs,
                                                           Good_Channels=self.Good_Channels,
                                                           Num_Freq=self.Num_Freq, SN_L=self.Sn_Len, Gp_L=self.Gap_Len)
        self.Silence_Neural, self.Silence_CAR = RR_Neural_Master(self.Silence_Neural, Num_Trials=self.Num_Silence,
                                                                 Good_Channels=self.Good_Channels,
                                                                 Num_Freq=self.Num_Freq, SN_L=self.Sn_Len,
                                                                 Gp_L=self.Gap_Len)
        self.Log_String = 'Re-Referenced with Common Average Reference Excluding Channel(s): %s' % self.Bad_Channels  # Construct Log String

    @_StandardStep
    def Z_Score(self):
        ''' Z-Score Input Data based on Equal Number of Song and Silence Trials
        '''
        ### !!!!!! Validate Proper Steps have been made made!!!!!!!
        self.Song_Neural, self.Silence_Neural, self.Means, self.StdDevs = Z_Score_data_Master(
            Frequencies_Song=self.Song_Neural, Frequencies_Silence=self.Silence_Neural, Numb_Freq=self.Num_Freq,
            Numb_Motifs=self.Num_Motifs, Numb_Silence=self.Num_Silence)
        self.Log_String = 'Z-Scored Data [Eqn: z = (x – μ) / σ]'  # Construct Log String

################### LAST WORKING HERE: Clean Up Documentation, Unit Test, Back-up on Github, and Work on Analysis


