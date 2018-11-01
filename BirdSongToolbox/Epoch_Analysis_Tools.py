"""
Epoch_Analysis_Tools
Tools for analyzing the predictive power of LFP and SUA for predicting Syllable onset

"""
import pickle

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

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


# TODO: Check out what is Happening in the Below Deprecation Warning
# /home/debrown/anaconda2/lib/python2.7/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module
#  was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and
# functions are moved. Also note that the interface of the new CV iterators are different from that of this module.
# This module will be removed in 0.20. #   "This module will be removed in 0.20.", DeprecationWarning)
# TODO: Check out what is Happening in the Above Deprecation Warning

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


def Get_LFP_Templates(Trials, Tr_Len, Gap_Len, Buffer):
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
def Get_Validation_Test_Sets(Epoch_Index):
    """Breaks Epochs of a certain day into a Validation and Test Set

    Parameters:
    -----------
    Epoch_Index: list
        List of all Epochs to be used (Typically the All_First_Motifs Epochs)

    Returns:
    --------
    validation_set: list
        Index of Epochs reserved for the Validation Set
    test_set: list
        Index of the Epochs to be used for K-Fold Cross Validation Testing
    """

    test_set, validation_set = train_test_split(Epoch_Index, random_state=0)

    return validation_set, test_set


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

    Channel_Full_Freq_Trials = []

    for Channel in range(0, D):  # Over all Channels
        Freq_Trials = []
        Freq_Full_Trials = []
        for l in range(0, len(TOP)):  # For Range of All Frequency Bins
            Chan_Full_Holder = np.zeros((4500, 1))  # Initiate Holder for Trials (Motifs)
            for motif in Sel_Motifs:  # For each value of Sel_Motifs
                Current_Full_Motif = Selected_Feature_Type[motif][Channel][:, l]  # Select[Motif][Ch][Epoch, Freq]

                # Line Up all of the Selected Frequencies across All Trials for that Channel
                Chan_Full_Holder = np.column_stack((Chan_Full_Holder, Current_Full_Motif))
            Chan_Full_Holder = np.delete(Chan_Full_Holder, 0, 1)  # Delete the First Column (Initialized Column)
            Freq_Full_Trials.append(Chan_Full_Holder)  # Save all of the Trials for that Frequency on that Channel
        Channel_Full_Freq_Trials.append(Freq_Full_Trials)

    return Channel_Full_Freq_Trials


# TODO: Build Test for Full_Trial_LFP_Clipper  here is a template: Pipe_1.Song_Neural[0][0][:,0] == Dataset[0][0][:,0]

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
                Current_Full_Motif = Neural[motif][Channel][:, Freq]  # Select[Motif][Ch][Epoch, Freq]

                # Line Up all of the Selected Frequencies across All Trials for that Channel
                Chan_Full_Holder = np.column_stack((Chan_Full_Holder, Current_Full_Motif))
            Chan_Full_Holder = np.delete(Chan_Full_Holder, 0, 1)  # Delete the First Column (Initialized Column)
            Freq_Full_Trials.append(Chan_Full_Holder)  # Save all of the Trials for that Frequency on that Channel
        Channel_Full_Freq_Trials.append(Freq_Full_Trials)

    return Channel_Full_Freq_Trials


########################################################################################################################
# Functions for Handling the new Dictionary Format of the Hand labels


def get_hand_labels(bird_id='z020', sess_name='day-2016-06-03', supp_path=None, local=False):
    """Function Imports the Hand Labels of the Specified Recording

    Inputs:
    -------
    bird_id: str
        Bird Identification; defaults to 'z020'
    sess_name: sstr
        Recording Date; defaults to 'day-2016-06-02',
    supp_path: str
        ; defaults to None

    Returns:
    --------
    Epoch_Dict: dict (If Save = False)
        Dictionary of the Handlabels; {Epoch_Numb : (Labels, Onsets)}
                                        |-> Labels: [Event_Labels]
                                        |-> Onsets: [[Starts], [Ends]]
    """
    # TODO: Make this function more reasonable
    if local == False:
        folder_path = '/home/debrown/Handlabels'
    elif local:
        assert isinstance(supp_path, str), "Supplemental path must be a String"
        folder_path = os.path.join(supp_path, bird_id, sess_name)
    else:
        folder_path = os.path.join(bird_id, sess_name)
        print('Here')
    file_name = bird_id + '_' + sess_name + '.pckl'
    Destination = os.path.join(folder_path, file_name)

    print(Destination)
    file = open(Destination, 'rb')
    Epoch_Dict = pickle.load(file)
    file.close()
    return Epoch_Dict


def Prep_Handlabels_for_ML(Hand_labels, Index):
    """Restructures the Dictionary of Hand labels to a list for use in Machine Learning Pipeline

    Parameters:
    -----------
    Hand_labels: dict
        Dictionary of the Handlabels; {Epoch_Numb : (Labels, Onsets)}
                                        |-> Labels: [Event_Labels]
                                        |-> Onsets: [[Starts], [Ends]]
    Index: list
        Index of Epochs to grab from the Handlabels dictionary

    Returns:
    --------
    labels_list: list
        [Epoch] -> [Labels]
    onsets_list: list
        [[Epochs]->[Labels] , [Epochs]->[Start Time]]

    """

    labels_list = []
    onsets_list = []
    starts = []
    ends = []

    for epoch in Index:
        labels_list.append(Hand_labels[epoch][0])
        starts_temp, ends_temp = Hand_labels[epoch][1]
        starts.append(starts_temp)
        ends.append(ends_temp)
    onsets_list.append(starts)
    onsets_list.append(ends)
    return labels_list, onsets_list


########################################################################################################################
##Label Handling Functions


def Label_Focus(Focus, Labels, Starts):
    """ Create a list of every instance of the User defined User Label (Focus on One Label)

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
    """Group Selected Labels together into One Label e.g. Combine Calls and Intro. Notes (Group these labels together)

    """
    Label_Index = []

    for i in range(len(Labels)):
        Group_Labels = []
        for j in range(len(Focuses)):
            Trial_Labels = [int(Starts[i][x] / 30) for x in range(len(Labels[i])) if Labels[i][x] == Focuses[j]]
            Group_Labels.extend(Trial_Labels)
        Label_Index.append(Group_Labels)
    return Label_Index


# Function for grabing more examples from a onset


def Label_Extract_Pipeline(Full_Trials, All_Labels, Starts, Label_Instructions, Offset=int, Tr_Length=int,
                           Slide=None, Step=False):
    """Extracts all of the Neural Data Examples of User Selected Labels and return them in the designated manner.

    Label_Instructions = tells the Function what labels to extract and whether to group them together

    Parameters:
    -----------
    Full_Trials: list
        [Ch]->[Freq]->(Time Samples x Trials)
    All_Labels: list
        List of all Labels corresponding to each Epoch in Full_Trials
        [Epochs]->[Labels]
    Starts: list
        List of all Start Times corresponding to each Epoch in Full_Trials
        [Epochs]->[Start Time]
    Label_Instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Offset = int
        The number of samples away from the true onset to Grab for ML Trials (Can be Before or After)
    Tr_Length=int
        Number of Samples to use for Features
    Slide: bool
        defaults to None

    Step:
        defaults to False

    Returns:
    -------
    clippings: list
        List containing the Segments Designamted by the Label_Instructions, Offset, and Tr_Length parameters
        [labels] -> [ch] -> [freq] -> ( Samples x Label Instances)
    templates: list
        List containing the mean across instances of each label for each Channel's Frequency Band
        [Labels]->[Ch]->[Freq]-> (Mean of Samples Accross Insances x 1)

    """

    clippings = []
    templates = []

    for instruction in range(len(Label_Instructions)):
        if type(Label_Instructions[instruction]) == int or type(Label_Instructions[instruction]) == str:
            label_starts = Label_Focus(Label_Instructions[instruction], All_Labels, Starts)
        else:
            label_starts = Label_Grouper(Label_Instructions[instruction], All_Labels, Starts)

        if type(Slide) == int:
            label_starts = Slider(label_starts, Slide=Slide, Step=Step)

        clips, temps = Dyn_LFP_Clipper(Full_Trials, label_starts, Offset=Offset, Tr_Length=Tr_Length)
        clippings.append(clips)
        templates.append(temps)
    return clippings, templates


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
            Chan = Features[Channel]  # Select Channel ##### Need to Change Channel to Channel Index (For For Loop)
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


def Dyn_LFP_Clipper(Features: list, Starts, Offset=int, Tr_Length=int):
    """This Function Dynamically clips Neural data prior to a selected label and re-organizes them for future use.

    Information:
    ------------
        **This function assumes that the Data has Been limited to only Epoch that will be Trained or Tested**
        It iterates over EACH Epoch clipping ALL examples of ONE label in each trial.
        It should be run repeatedly for clipping all of the designated labels.

        Its output can later be broken into an Initial Training and Final Test Set. [May Not Be True]

        Note: The Number of Examples of Labels does not always equal the total number of examples total as some push
        past the timeframe of the Epoch and are excluded

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

    Dynamic_Templates: list
        List of Stacked Template Neural Data that corresponds to the Label designated (Templates are the mean of trials)
        [Labels]-> [Ch]->[Freq]->(Time (Samples) x 1)

    """

    num_chan = len(Features[:])  # Number of Channels
    freq_bands = len(Features[0][:])  # Num of Frequency Bands
    num_trials = len(Features[0][0][0, :])  # Number of Trials of Dynam. Clipped Training Set
    num_examples = Numel(Starts) - Numbad(Starts, Offset=Offset, Tr_Length=Tr_Length) - Numbad2(Starts, len(
        Features[0][0][:, 0]), Offset=Offset)  # Number of Examples

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
                    if Starts[epoch][example] - Offset - Tr_Length >= 0 and Starts[epoch][example] - Offset < len(
                            sel_freq_epochs):
                        # if len(sel_freq_epochs[Starts[epoch][example] - Offset - Tr_Length:Starts[epoch][example] - Offset, epoch]) == 9:
                        #     print(Starts[epoch][example] - Offset - Tr_Length)
                        #     print(Starts[epoch][example] - Offset)  # Select Motif)
                        Chan_Holder[:, Counter] = sel_freq_epochs[
                                                  Starts[epoch][example] - Offset - Tr_Length:Starts[epoch][
                                                                                                  example] - Offset,
                                                  epoch]  # Select Motif
                        Counter = Counter + 1
            Freq_Trials.append(Chan_Holder)  # Save all of the Trials for that Frequency on that Channel
            Chan_Means = np.mean(Chan_Holder, axis=1)  # Find Means (Match Filter)
            # Store all Match Filters for Every Frequency for that Channel
            Matches.append(Chan_Means.reshape(Tr_Length, 1))
        Dynamic_Templates.append(Matches)
        Dynamic_Freq_Trials.append(Freq_Trials)  # Save all of the Trials for all Frequencies on each Channel

    return Dynamic_Freq_Trials, Dynamic_Templates


########################################################################################################################


def Find_Power(Features, Pow_Method='Basic'):
    """ Function to Find the Power for all Trials (Intermediate Preprocessing Step)

    Parameters:
    -----------
    Features: list
        List containing all the Segments Clipped  for one labels.
        (As defined by Label_Instructions in Label_Extract_Pipeline)
        [ch] -> [freq] -> ( Samples x Label Instances)
    Pow_Method: str
        Method by which power is taken (Options: 'Basic': Mean , 'MS': Mean Squared, 'RMS': Root Mean Square)

    Returns:
    --------
    Power_Trials: list
        [ch] -> [freq] -> ( Mean of Samples x Label Instances)
                            **(1 x Num Label Instances)
    """
    # Create Variable for IndexingF
    num_trials = len(Features[0][0][0, :])  # Number of Instances of Dynam. Clipped Training Set (Labels)

    # Create Lists
    power_trials = []

    for channel in Features[:]:  # Over all Channels
        freq_trials = []
        for frequency in channel[:]:  # For Range of All Frequency Bins
            #             print abs(Features[Channel - 1][l])
            if Pow_Method == 'Basic':
                chan_holder = np.average(abs(frequency), axis=0)
            if Pow_Method == 'MS':
                chan_holder = np.mean(np.power(frequency, 2), axis=0)
            if Pow_Method == 'RMS':
                chan_holder = np.power(np.mean(np.power(frequency, 2), axis=0), .5)

            chan_holder = np.reshape(chan_holder, (num_trials, 1))
            freq_trials.append(chan_holder)  # Save all of the Trials for that Frequency on that Channel
        power_trials.append(freq_trials)  # Save all of the Trials for all Frequencies on each Channel
    return power_trials


def Pearson_Coeff_Finder(Features, Templates):
    """ This Function Mirrors Power_Finder only for finding Pearson Correlation Coefficient
    It iterates over each Template and finds the Pearson Coefficient for 1 template at a time

    Information:
    ------------
            Note: The Number of Examples of Label does not always equal the total number of examples total as some push past
            the time frame of the Epoch and are excluded

    Parameters:
    -----------
    Features: list
        List containing all the Segments Clipped  for one labels.
        (As defined by Label_Instructions in Label_Extract_Pipeline)
        [ch] -> [freq] -> ( Samples x Label Instances)
    Templates: list
        List of Stacked Template Neural Data that corresponds to the Label designated (Templates are the mean of trials)
        [Labels]->[Ch]->[Freq]->(Time (Samples) x 1)

    Returns:
    --------
    corr_trials: list
        list of Pearson Correlation Values between each instance and the LFP Template of each Label
        [ch] -> [freq] -> ( Number of Instances x Number of Labels/Templates)

    """

    # Create Variable for IndexingF
    num_trials = len(Features[0][0][0, :])  # Number of Trials of Dynam. Clipped Training Set
    num_temps = len(Templates)

    # Create Lists
    corr_trials = []

    for channel in range(0, len(Features[:])):  # Over all Channels
        freq_trials = []
        for frequency in range(len(Features[0][:])):  # For Range of All Frequency Bins
            corr_holder = np.zeros([num_trials, num_temps])

            for instance in range(num_trials):
                for temp in range(num_temps):
                    corr_holder[instance, temp], _ = scipy.stats.pearsonr(Features[channel][frequency][:, instance],
                                                                          Templates[temp][channel][frequency][:, 0])
            freq_trials.append(corr_holder)  # Save all of the Trials for that Frequency on that Channel
        corr_trials.append(freq_trials)  # Save all of the Trials for all Frequencies on each Channel
    return corr_trials


def Pearson_Extraction(Clipped_Trials, Templates):
    """

    Parameters:
    -----------
    Clipped_Trials: list
        List containing the Segments Clipped by Label_Extract_Pipeline
        [labels] -> [ch] -> [freq] -> ( Samples x Label Instances)
    Templates: list
        List of Stacked Template Neural Data that corresponds to the Label designated (Templates are the mean of trials)
        [Labels]->[Ch]->[Freq]->(Time (Samples) x 1)

    Returns:
    --------
    Extracted_Pearson: list
        list of Pearson Correlation Values between each instance and the LFP Template of each Label
        [Labels]->[ch] -> [freq] -> ( Number of Instances x Number of Labels/Templates)
    """
    Extracted_Pearson = []
    for i in range(len(Clipped_Trials)):
        Extracted_Pearson.append(Pearson_Coeff_Finder(Clipped_Trials[i], Templates=Templates))
    return Extracted_Pearson


# Function for getting the Pearson Coefficient for Classification
def Pearson_ML_Order(Features):
    """Reorganizes the Extracted Features into a Useful Machine Learning Format

    Output Shape [Number of Examples vs. Number of Features]

    """
    # Create Variable for Indexing
    #     Num_Temps = len(Features[0][0][0,:]) # Number of Templates
    NT = len(Features[0][0][:, 0])  # Number of Trials
    # Create Initial Array
    column_index = []
    ordered_trials = np.zeros((NT, 1))  # Initialize Dummy Array

    # Channel Based Ordering
    for channel in Features:  # Over all Channels
        for frequency in channel:  # For Range of All Frequency Bins
            for temps in range(len(frequency[0, :])):
                ordered_trials = np.concatenate((ordered_trials, np.reshape(frequency[:, temps], (NT, 1))), axis=1)
                universal_index = (channel, frequency, temps)  # Tuple that contains (Channel #, Freq Band #)
                column_index.append(universal_index)  # Append Index Tuple in Column Order
    ordered_trials = np.delete(ordered_trials, 0, 1)  # Delete the First Row (Initialized Row)
    return ordered_trials, column_index


def Pearson_ML_Order_Pipeline(Extracted_Features):
    """

    :param Extracted_Features:
    :return:

    """
    ML_Ready = np.zeros((1, (len(Extracted_Features[0]) * len(Extracted_Features[0][0]) * len(Extracted_Features))))
    ML_Labels = np.zeros((1, 1))
    for i in range(len(Extracted_Features)):
        Ordered_Trials, Ordered_Index = Pearson_ML_Order(Extracted_Features[i])
        ML_Ready = np.concatenate((ML_Ready, Ordered_Trials), axis=0)

        # Handles Labels so they are flexible when grouping
        ROW, COLL = np.shape(Ordered_Trials)
        Dyn_Labels = np.zeros([ROW, 1])
        Dyn_Labels[:, 0] = i
        ML_Labels = np.concatenate((ML_Labels, Dyn_Labels), axis=0)

    ML_Ready = np.delete(ML_Ready, 0, 0)
    ML_Labels = np.delete(ML_Labels, 0, 0)
    return ML_Ready, ML_Labels, Ordered_Index


def Power_Extraction(Clipped_Trials):
    """

    Parameters:
    -----------
    Clipped_Trials: list
        List containing the Segments Clipped by Label_Extract_Pipeline
        [labels] -> [ch] -> [freq] -> ( Samples x Label Instances)

    Return:
    -------
    extracted_power: list
        [labels] -> [ch] -> [freq] -> ( Mean of Samples x Label Instances)
                                        (1 x Num Label Instances)
    """
    extracted_power = []
    for label in Clipped_Trials:
        extracted_power.append(Find_Power(label))
    return extracted_power


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


def Select_Classifier(Model=str, Strategy=str):
    if Model == 'GNB':
        classifier = GaussianNB()
    if Model == 'LDA':
        classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    if Strategy == '1vAll':
        classifier = OneVsRestClassifier(classifier)
    return classifier


# Functions for Ordering Features into useful Feature Drop Format
# Need to add Function to Selectively Drop Frequencies
## Made Corrections on 10/27/2017 additional ones on 10/30/2017

def ML_Order(Features):
    '''Reorganizes the Extracted Features into a Useful Machine Learning Format

    Parameters:
    -----------

    Returns:
    --------
    Ordered_Trials:

    Column_Index:
        ?????(Ch, freq_trials)???? Not Sure!
        Output Shape [Number of Examples vs. Number of Features]
    '''
    # Create Variable for Indexing
    #     D = len(Features[:]) # Number of Channels
    B = len(Features[0][0][:, 0])  # Length of Dynam. Clipped Training Set
    #     NT = len(Features[0][0][0,:]) # Number of Trials of Dynam. Clipped Training Set

    # Create Initial Array
    Column_Index = []
    Ordered_Trials = np.zeros((B, 1))  # Initialize Dummy Array

    # Channel Based Ordering
    for Channel in range(0, len(Features)):  # Over all Channels
        Corr_Trials = []  # Create Dummy Lists
        for l in range(len(Features[0][:])):  # For Range of All Frequency Bins
            Current_Feature = Features[Channel][l]
            Ordered_Trials = np.concatenate((Ordered_Trials, Current_Feature), axis=1)
            Tuple = (Channel, l)  # Tuple that contains (Channel #, Freq Band #)
            Column_Index.append(Tuple)  # Append Index Tuple in Column Order

    Ordered_Trials = np.delete(Ordered_Trials, 0, 1)  # Delete the First Row (Initialized Row)

    return Ordered_Trials, Column_Index


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


def Key_Operator(Column_Key):
    """Function for Organizing Channel/Frequency specific Dropping

    :param Column_Key:
    :return:
    """
    CH, FREQ = max(Column_Key)
    CH = CH + 1
    FREQ = FREQ + 1
    # Work for Pulling out Useful Keys
    Chan_Index = {}
    Freq_Index = {}

    for i in range(CH):
        Chan_Holder = []
        for k in range(len(Column_Key)):
            h, p = Column_Key[k]
            if h == i:
                Chan_Holder.append(k)
        Chan_Index[i] = Chan_Holder

    for j in range(FREQ):
        Freq_Holder = []
        for k in range(len(Column_Key)):
            h, p = Column_Key[k]
            if p == j:
                Freq_Holder.append(k)
        Freq_Index[j] = Freq_Holder

    return Chan_Index, Freq_Index


def Drop_Features(Features, Keys, Desig_Drop):
    """Function for Selectively Removing Columns for Feature Dropping
    Des_igDrop is short for Designated to be Dropped"""
    Full_Drop = []

    for i in range(len(Desig_Drop)):
        Full_Drop.extend(Keys[Desig_Drop[i]])

    Remaining_Features = np.delete(Features, Full_Drop, axis=1)

    return Remaining_Features, Full_Drop


# TODO: Machine_Learning_Prep needs to be re-evaluated. Particularly it may need to be done after taking a Validation Set
def Machine_Learning_PreP(Song_Trials, Silence_Trials, verbose=False):
    """Determine Number of Examples of Song and Silence

    """
    num_Songs = len(Song_Trials[:, 0])  # Number of Examples of Song
    num_Silences = len(Silence_Trials[:, 0])  # Number of Examples of Silence

    Song_Labels = np.ndarray([num_Songs])  # Make ndarray
    Song_Labels[:] = 1
    Silence_Labels = np.ndarray([num_Silences])
    Silence_Labels[:] = 0

    K = 4

    # Prep Data for K-Fold Function
    X = np.concatenate((Song_Trials, Silence_Trials), axis=0)
    y = np.concatenate((Song_Labels, Silence_Labels), axis=0)  # Create Label Array

    # X = np.transpose(X)
    # y = np.array([0, 0, 0])
    skf = StratifiedKFold(y, n_folds=K)

    return X, y, skf


########################################################################################################################
################################ Code for Handling Full Series Operations ##############################################
########################################################################################################################

# This function is the Second half of Run_GNB
# It is broken in half to cut down time for computation (K-fold and Label creation only done once)

from sklearn import metrics


def KFold_Classification(Data_Set, Data_Labels, Method='GNB', Strategy='1vALL',
                         verbose=False):  # , SKF, verbose=False):
    ''' This Function is a Flexible Machine Learning Function that Trains FOUR Classifiers and determines metrics of each
    The metrics it determines are:
                [1] Accuracy & StdERR
                [2] Confusion Matrix
                [3] Reciever Operating Curve
    It stores all of the trained Classifiers into a Dictionary and pairs it with a
    Dictionary of the Corresponding Test Index. The Pair are returned as a tuple'''

    K = 4
    class_index = list(set(Data_Labels))
    X = Data_Set
    y = Data_Labels

    #     y = label_binarize(Data_Labels, classes=list(set(Data_Labels)))

    skf = StratifiedKFold(Data_Labels, n_folds=K)

    acc = np.zeros(K)
    c = []  # Just Added
    ROC = []  # Just Added too 8/10
    foldNum = 0

    Trained_Classifiers = dict()
    Trained_Index = dict()
    for train, test in skf:
        if verbose:
            print("Fold %s..." % foldNum)
        # print "%s %s" % (train, test)
        X_train, y_train = X[train, :], y[train]
        X_test, y_test = X[test, :], y[test]

        Classifier = Select_Classifier(Model=Method, Strategy=Strategy)
        Classifier.fit(X_train, y_train)

        y_pred = Classifier.predict(X_test)

        C = confusion_matrix(y_test, y_pred).astype(float)
        numTestTrials = len(y_test)
        acc[foldNum] = sum(np.diag(C)) / numTestTrials
        foldNum += 1

        Trained_Classifiers[foldNum - 1] = Classifier
        Trained_Index[foldNum - 1] = test

        if verbose:
            print(C)
        c.append(C)
    #         ROC.append(roc_holder)

    meanAcc_nb = np.mean(acc)
    stdErr_nb = np.std(acc) / np.sqrt(K)
    Classifier_Components = (Trained_Classifiers, Trained_Index)

    if verbose:
        print("cross-validated acc: %.2f +/- %.2f" % (np.mean(acc), np.std(acc)))

    return meanAcc_nb, stdErr_nb, Classifier_Components, c


def Make_Full_Trial_Index(Features, Offset=int, Tr_Length=int):
    FT_Index = []
    for i in range(len(Features[0][0][0, :])):
        Time_List = np.arange(Offset + Tr_Length, 4500)
        FT_Index.append(list(Time_List))
    return FT_Index


# Function needs to be able to Selectively choose which Trials to run full Trial
def Series_LFP_Clipper(Features, Offset=int, Tr_Length=int):
    """This Function Sequentially clips Neural data attributed for a full trial and organizes
    them for future steps. It iterates over EACH full trial clipping.
    It should be run once.

    Its output can later be broken into an Initial Training and Final Test Set.


    Starts is a list of Lists containing the Start Times of only One type of Label in each Clipping.
    Also Note that the Starts Argument must be converted to the 1 KHz Sampling Frequency

    Offset = How much prior or after onset
    """

    ### Consider removing the Create_Bands Step and consider using len()
    ### ESPECIALLY IF YOU CHANGE THE BANDING CODE

    Starts = Make_Full_Trial_Index(Features, Offset=Offset, Tr_Length=Tr_Length)

    D = len(Features[:])  # Number of Channels
    F = len(Features[0][:])  # Num of Frequency Bands
    NT = len(Features[0][0][0, :])  # Number of Trials of Dynam. Clipped Training Set
    NEl = Numel(Starts) - Numbad(Starts, Offset=Offset, Tr_Length=Tr_Length)  # Number of Examples

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
                    if Starts[Trials][Ex] - Offset - Tr_Length >= 0:
                        Chan_Holder[:, Counter] = Freq[
                                                  Starts[Trials][Ex] - Offset - Tr_Length:Starts[Trials][Ex] - Offset,
                                                  Trials]  # Select Motif
                        Counter = Counter + 1
            Freq_Trials.append(Chan_Holder)  # Save all of the Trials for that Frequency on that Channel
        Dynamic_Freq_Trials.append(Freq_Trials)  # Save all of the Trials for all Frequencies on each Channel

    return Dynamic_Freq_Trials


# Clip_Test, Temp_Test = Label_Extract_Pipeline(Dataset,  Stereotype_Labels, Stereotype_Clippings[0],  [1,2,3,4], Offset = 10, Tr_Length= 20)
# Power_List = Power_Extraction(Clip_Test)
# ML_Trial_Test, ML_Labels_Test, Ordered_Index_TEST = ML_Order_Pipeline(Power_List)


def Classification_Prep_Pipeline(Full_Trials, All_Labels, Time_Stamps, Label_Instructions, Offset=int, Tr_Length=int,
                                 Feature_Type=str, Temps=None, Slide=None, Step=False):
    Clips, Temps_internal = Label_Extract_Pipeline(Full_Trials,
                                                   All_Labels,
                                                   Time_Stamps,
                                                   Label_Instructions,
                                                   Offset=Offset,
                                                   Tr_Length=Tr_Length,
                                                   Slide=Slide,
                                                   Step=Step)

    if Feature_Type == 'Power':
        Power = Power_Extraction(Clips)
        ML_Trials, ML_Labels, Ordered_Index = ML_Order_Pipeline(Power)

    if Feature_Type == 'Pearson':
        print
        'Probably need to Validate Pearson Functions Correctly'
        # Function for Finding Pearson
        ## [Probably should add *kwargs]
        # Fucntion for Ordering Pearson
        if Temps == None:
            Pearson = Pearson_Extraction(Clips, Temps_internal)
        if Temps != None:
            Pearson = Pearson_Extraction(Clips, Temps)

        ML_Trials, ML_Labels, Ordered_Index = Pearson_ML_Order_Pipeline(Pearson)

        if Temps == None:
            return ML_Trials, ML_Labels, Ordered_Index, Temps_internal

    return ML_Trials, ML_Labels, Ordered_Index


def Series_Classification_Prep_Pipeline(Features, Offset=int, Tr_Length=int, Feature_Type=str, Temps=None):
    Series_Trial = Series_LFP_Clipper(Features, Offset=Offset, Tr_Length=Tr_Length)

    if Feature_Type == 'Power':
        Series_Power = Find_Power(Series_Trial)
        Series_Ordered, _ = ML_Order(Series_Power)

    if Feature_Type == 'Pearson':
        print
        'Under Development'
        # Function for Finding Pearson
        ## [Probably should add *kwargs]
        # Fucntion for Ordering Pearson
        Series_Pearson = Pearson_Coeff_Finder(Series_Trial, Temps)
        Series_Ordered, _ = Pearson_ML_Order(Series_Pearson)

    Full_Trial_Features = []

    Trial_Length = len(Features[0][0][:, 0]) - Offset - Tr_Length

    for i in range(len(Features[0][0][0, :])):
        Full_Trial_Features.append(Series_Ordered[Trial_Length * (i):Trial_Length * (i + 1), :])

    return Full_Trial_Features


def KFold_Series_Prep(Data_Set, Test_index, Offset=int, Tr_Length=int, Feature_Type=str):
    Trial_set = Trial_Selector(Features=Data_Set, Sel_index=Test_index)

    series_ready = Series_Classification_Prep_Pipeline(Trial_set, Offset=Offset, Tr_Length=Tr_Length,
                                                       Feature_Type=Feature_Type)
    return series_ready


## Function Only Works with SciKitLearn 19.1 and later [Due to Change in how skf syntax]
# Need a KFold Classification method that folds Full Clippings so that I can test on a fresh Psuedal Training Set

# Steps:
# [Initialization]:
# [1] Specify Classifier Model and Parameters
# [Probably just pass Function with desired Parameters and return object]
# [2] Number of Folds, [3] Clippings with there
# Steps:
# [1] Break Clippings into K-Folds
# [2] For Each K-Fold
# [2.1] Extract and Order the Designated Features using Perscribed Methods
# [2.2] Train Model
# [2.3] Test Model
# [2.4] Get Confusion Matrix
# [2.5] Get Accuracy and Std
# [2.7] Store Trained Model
# [2.6] Return: Confusion Matrix, Accuracy, Std, Trained Model
# [3] Store Results into dictionary(s) also store index of the Test Set Left Out


def Clip_Classification(Class_Obj, Train_Set, Train_Labels, Test_Set, Test_Labels, verbose=False):
    ''' This Function is a Flexible Machine Learning Function that Trains One Classifier and determines metrics for it
    The metrics it determines are:
                [1] Accuracy & StdERR
                [2] Confusion Matrix
                [3] Reciever Operating Curve
    It stores all of the trained Classifiers into a Dictionary and pairs it with a
    Dictionary of the Corresponding Test Index. The Pair are returned as a tuple'''

    Classifier = Class_Obj
    Classifier.fit(Train_Set, Train_Labels)

    Test_pred = Classifier.predict(Test_Set)

    C = confusion_matrix(Test_Labels, Test_pred).astype(float)
    numTestTrials = len(Test_Labels)

    acc = sum(np.diag(C)) / numTestTrials
    return acc, Classifier, C


def Trial_Selector(Features, Sel_index):
    '''This Function allows you to easily parse our specific Trials for K-Fold validation
    and Test Set Seperation'''
    num_chans, num_freqs, Clip_len, _ = np.shape(Features)
    Num_trials = len(Sel_index)

    Sel_Trials = []

    for i in range(num_chans):
        freq_holder = []
        for j in range(num_freqs):
            trial_holder = np.zeros([Clip_len, Num_trials])
            for k in range(len(Sel_index)):
                trial_holder[:, k] = Features[i][j][:, Sel_index[k]]
            freq_holder.append(trial_holder)
        Sel_Trials.append(freq_holder)
    return Sel_Trials


def Label_Selector(Labels, Sel_index):
    '''This Function allows you to easily parse our specific Trial's Labels for K-Fold validation
    and Test Set Seperation'''
    Sel_Labels = []
    for i in xrange(len(Sel_index)):
        Sel_Labels.append(Labels[Sel_index[i]])
    return Sel_Labels


def Convienient_Selector(Features, Labels, Starts, Sel_index):
    sel_set = Trial_Selector(Features=Features, Sel_index=Sel_index)
    sel_labels = Label_Selector(Labels, Sel_index=Sel_index)
    sel_starts = Label_Selector(Starts, Sel_index=Sel_index)
    return sel_set, sel_labels, sel_starts


#### NEED TO AD OPTIONAL IF STATETMENT HANDLING FOR RETURNING THE TEMPLATES FOR SERIES CLASSIFICATION

def Clip_KFold(Class_Obj, Data_Set, Data_Labels, Data_Starts, Label_Instructions, Offset=int, Tr_Length=int,
               Feature_Type=str, K=4, Slide=None, Step=False, verbose=False):
    #     Data_Set, Data_Labels, Data_Starts, Label_Instructions, Offset = int, Tr_Length= int, Feature_Type = str) , Temps = None

    skf = StratifiedKFold(n_splits=K)

    acc = np.zeros(K)
    c = []  # Just Added
    ROC = []  # Just Added too 8/10
    foldNum = 0

    Trained_Classifiers = dict()
    Trained_Index = dict()

    Num_Clippings = np.ones(len(Data_Labels))

    for train, test in skf.split(Num_Clippings, Num_Clippings):
        if verbose:
            print("Fold %s..." % foldNum)
            # print "%s %s" % (train, test)

        print(train)
        train_set, train_labels, train_starts = Convienient_Selector(Data_Set, Data_Labels, Data_Starts, train)

        print(test)
        test_set, test_labels, test_starts = Convienient_Selector(Data_Set, Data_Labels, Data_Starts, test)

        if Feature_Type != 'Pearson':
            ML_Train_Trials, ML_Train_Labels, Train_Ordered_Index = Classification_Prep_Pipeline(train_set,
                                                                                                 train_labels,
                                                                                                 train_starts,
                                                                                                 Label_Instructions,
                                                                                                 Offset=Offset,
                                                                                                 Tr_Length=Tr_Length,
                                                                                                 Feature_Type=Feature_Type,
                                                                                                 Temps=None,
                                                                                                 Slide=Slide,
                                                                                                 Step=Step)

            ML_Test_Trials, ML_Test_Labels, Test_Ordered_Index = Classification_Prep_Pipeline(test_set,
                                                                                              test_labels,
                                                                                              test_starts,
                                                                                              Label_Instructions,
                                                                                              Offset=Offset,
                                                                                              Tr_Length=Tr_Length,
                                                                                              Feature_Type=Feature_Type,
                                                                                              Temps=None,
                                                                                              Slide=Slide,
                                                                                              Step=Step)

        if Feature_Type == 'Pearson':
            ML_Train_Trials, ML_Train_Labels, Train_Ordered_Index, Temps_int = Classification_Prep_Pipeline(train_set,
                                                                                                            train_labels,
                                                                                                            train_starts,
                                                                                                            Label_Instructions,
                                                                                                            Offset=Offset,
                                                                                                            Tr_Length=Tr_Length,
                                                                                                            Feature_Type=Feature_Type,
                                                                                                            Temps=None,
                                                                                                            Slide=Slide,
                                                                                                            Step=Step)

            ML_Test_Trials, ML_Test_Labels, Test_Ordered_Index = Classification_Prep_Pipeline(test_set,
                                                                                              test_labels,
                                                                                              test_starts,
                                                                                              Label_Instructions,
                                                                                              Offset=Offset,
                                                                                              Tr_Length=Tr_Length,
                                                                                              Feature_Type=Feature_Type,
                                                                                              Temps=Temps_int,
                                                                                              Slide=Slide,
                                                                                              Step=Step)

        acc[foldNum], Trained_Classifiers[foldNum], C = Clip_Classification(Class_Obj, ML_Train_Trials, ML_Train_Labels,
                                                                            ML_Test_Trials, ML_Test_Labels,
                                                                            verbose=False)
        Trained_Index[foldNum] = test
        foldNum += 1

        if verbose:
            print(C)
        c.append(C)

    meanAcc_nb = np.mean(acc)
    stdErr_nb = np.std(acc) / np.sqrt(K)
    Classifier_Components = (Trained_Classifiers, Trained_Index)

    if verbose:
        print("cross-validated acc: %.2f +/- %.2f" % (np.mean(acc), np.std(acc)))
    return meanAcc_nb, stdErr_nb, Classifier_Components, c,
