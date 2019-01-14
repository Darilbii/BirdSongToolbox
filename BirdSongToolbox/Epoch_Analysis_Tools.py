"""
Epoch_Analysis_Tools
Tools for analyzing the predictive power of LFP and SUA for predicting Syllable onset

"""
import pickle

import scipy
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.preprocessing import label_binarize
import matplotlib.patches as mpatches

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
import copy


# The Following Function Finds the Template for Each Motif for Each Frequency Band on Each Channel
# Edited/Written 2/14/2018

def Mean_match(Features, Sel_Motifs, Num_Chan, Num_Freq, Sn_Len, Gap_Len, OffSet=0):
    """ Re-Organizes Data for Machine Learning and Visualization. It also Finds the Mean of Each Frequency Band on Each Channel

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
    """
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
    """Function grabs the time segment of Neural Activity during the designated time Vocal Behavior

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

    """

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
    """ Grabs every epoch in Sel_Motifs for Each Frequency Band on Each Channel and returns them in a structure list


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

    """
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
        [[Epochs]->[Start TIme] , [Epochs]->[End Time]]

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
    Labels: list
        List of all Labels corresponding to each Epoch in Full_Trials
        [Epochs]->[Labels]
    Starts: list
        List of all Start Times corresponding to each Epoch in Full_Trials
        [Epochs]->[Start Time]

    Returns:
    --------
    Label_Index: list
        List of all start frames of every instances of the label of focus
        [Num_Trials]->[Num_Exs]
    """
    Label_Index = []

    # for i in range(len(Labels)):
    #     Trial_Labels = [int(Starts[i][x] / 30) for x in range(len(Labels[i])) if Labels[i][x] == Focus]

    for start, epoch in zip(Starts, Labels):
        trial_labels = [int(start[i] / 30) for i, x in enumerate(epoch) if x == Focus]
        Label_Index.append(trial_labels)
    return Label_Index


# Function for Grouping Multiple Labels into 1 Label (e.g. Combine Calls and Introductory Notes)

def Label_Grouper(Focuses, Labels, Starts):
    """Group Selected Labels together into One Label e.g. Combine Calls and Intro. Notes (Group these labels together)

    """
    label_index = []

    for start, epoch in zip(Starts, Labels):
        group_labels = []
        for foc_label in Focuses:
            trial_labels = [int(start[i] / 30) for i, x in enumerate(epoch) if x == foc_label]
            group_labels.extend(trial_labels)
        label_index.append(group_labels)
    return label_index


# Function for grabing more examples from a onset


def Label_Extract_Pipeline(Full_Trials, All_Labels, Starts, Label_Instructions, Offset=int, Tr_Length=int,
                           Slide=None, Step=False):
    """Extracts all of the Neural Data Examples of User Selected Labels and return them in the designated manner.

    Label_Instructions = tells the Function what labels to extract and whether to group them together

    Parameters:
    -----------
    Full_Trials: list
        List of Full Epochs, this is typically output from Full_Trial_LFP_Clipper
        [Ch] -> [Freq] -> (Time Samples x Epochs)
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
        #TODO: Explain and Validate the Slide Parameter
    Step:
        defaults to False
        #TODO: Explain and Validate the Step Parameter

    Returns:
    -------
    clippings: list
        List containing the Segments Designated by the Label_Instructions, Offset, and Tr_Length parameters
        [labels] -> [ch] -> [freq] -> ( Samples x Label Instances)
    templates: list
        List containing the mean across instances of each label for each Channel's Frequency Band
        [Labels]->[Ch]->[Freq]-> (Mean of Samples across Instances  x  1)

    """

    clippings = []
    templates = []

    for instruction in range(len(Label_Instructions)):
        if type(Label_Instructions[instruction]) == int or type(Label_Instructions[instruction]) == str:
            label_starts = Label_Focus(Label_Instructions[instruction], All_Labels, Starts)
        else:
            label_starts = Label_Grouper(Label_Instructions[instruction], All_Labels, Starts)

        if type(Slide) == int:
            label_starts = Slider(label_starts, len(Full_Trials[0][0][:, 0]), Slide=Slide, Step=Step)

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


def Numbad(Index, Offset: int, Tr_Length: int):
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


def Numbad2(Index, ClipLen, Offset: int):
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
        (Note: that the Starts Argument must be converted to the 1 KHz Sampling Frequency)
        [Epochs]->[Start Time (For only one Type of Label)]
    Offset: int
        How much prior or after onset

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


def find_power(Features, Pow_Method='Basic'):
    """Finds the Power for all Instances of  One Label for every Channel's Frequency Bands (Modular Preprocessing Step)

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
        [ch] -> [freq] -> (Mean of Samples for each Label Instance  x  1 )
                            **(Num Label Instances x 1)**
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


def efficient_pearson_1d_v_2d(one_dim, two_dim):
    """Finds the Pearson correlation of all rows of the two dimensional array with the one dimensional array

    Source:
    -------
        https://www.quora.com/How-do-I-calculate-the-correlation-of-every-row-in-a-2D-array-to-a-1D-array-of-the-same-length

    Info:
    -----
        The Pearson correlation coefficient measures the linear relationship
     between two datasets. Strictly speaking, Pearson's correlation requires
     that each dataset be normally distributed. Like other correlation
     coefficients, this one varies between -1 and +1 with 0 implying no
     correlation. Correlations of -1 or +1 imply an exact linear
     relationship. Positive correlations imply that as x increases, so does
     y. Negative correlations imply that as x increases, y decreases.


    Parameters:
    ----------
    one_dim = ndarray
        1-Dimensional Array
    two_dim= ndarray
        2-Dimensional array it's row length must be equal to the length of one_dim

    Returns:
    --------
    pearsons: ndarray
    #TODO: Figure out the dimensions of this return


    Example:
    --------
    x = np.random.randn(10)
    y = np.random.randn(100, 10)

    The numerators is shape (100,) and denominators is shape (100,)
    Pearson = efficient_pearson_1d_v_2d(one_dim = x, two_dim = y)
    """
    x_bar = np.mean(one_dim)
    x_intermediate = one_dim - x_bar
    y_bar = np.mean(two_dim, axis=1)  # this flattens y to be (100,) which is a 1D array.
    # The problem is that y is 100, 10 so numpy's broadcasting doesn't know which axis to treat as the one to broadcast over.
    y_bar = y_bar[:, np.newaxis]
    # By adding this extra dimension, we're forcing numpy to treat the 0th axis as the one to broadcast over
    # which makes the next step possible. y_bar is now 100, 1
    y_intermediate = two_dim - y_bar
    numerators = y_intermediate.dot(x_intermediate)  # or x_intermediate.dot(y_intermediate.T)
    x_sq = np.sum(np.square(x_intermediate))
    y_sqs = np.sum(np.square(y_intermediate), axis=1)
    denominators = np.sqrt(x_sq * y_sqs)  # scalar times vector
    pearsons = (numerators / denominators)  # numerators is shape (100,) and denominators is shape (100,)

    return pearsons


def find_pearson_coeff(Features, Templates, Slow=False):
    """ Iterates over each Template and finds the Pearson Coefficient for 1 template at a time

        Note: This Function Mirrors find_power() only for finding Pearson Correlation Coefficient

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
    Slow: bool (Optional)
        if True the code will use the native scipy.stats.pearsonr() function which is slow (defaults to False)

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

    if Slow == True:
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

    else:
        for channel in range(len(Features[:])):  # Over all Channels
            freq_trials = []
            for frequency in range(len(Features[0][:])):  # For Range of All Frequency Bins
                corr_holder = np.zeros([num_trials, num_temps])
                for temp in range(num_temps):
                    corr_holder[:, temp] = efficient_pearson_1d_v_2d(Templates[temp][channel][frequency][:, 0],
                                                                     np.transpose(Features[channel][frequency]))
                freq_trials.append(corr_holder)  # Save all of the Trials for that Frequency on that Channel
            corr_trials.append(freq_trials)  # Save all of the Trials for all Frequencies on each Channel

    return corr_trials


def pearson_extraction(Clipped_Trials, Templates):
    """  Pearson Correlation Coefficients for all Labels

    Parameters:
    -----------
    Clipped_Trials: list
        List containing the Segments Clipped by Label_Extract_Pipeline
        [labels] -> [ch] -> [freq] -> ( Samples x Label Instances)
    Templates: list
        List of Stacked Template Neural Data that corresponds to the Label designated (Templates are the mean of trials)
        [Labels] -> [Ch] -> [Freq] -> (Time (Samples) x 1)

    Returns:
    --------
    Extracted_Pearson: list
        list of Pearson Correlation Values between each instance and the LFP Template of each Label
        [Labels]->[ch] -> [freq] -> ( Number of Instances  x  Number of Labels/Templates)
    """
    Extracted_Pearson = []
    for label in Clipped_Trials:
        Extracted_Pearson.append(find_pearson_coeff(label, Templates=Templates))
    return Extracted_Pearson


# Function for getting the Pearson Coefficient for Classification
def pearson_ml_module(Features):
    """Reorganizes the Extracted Pearson Features of One Lable's instance/samples into a Useful Machine Learning Format

    Output Shape [Number of Examples vs. Number of Features]

    Parameters:
    -----------
    Features: list
        list of Pearson Correlation Values between each instance and the LFP Template of each Label
        [ch] -> [freq] -> ( Number of Instances x Number of Labels/Templates)

    Returns:
    --------
    ordered_trials:
        [

    column_index:

    """
    # Create Variable for Indexing
    # NT = len(Features[0][0][:, 0])  # Number of Trials
    # Create Initial Array
    column_index = []
    ordered_trials = np.zeros((len(Features[0][0][:, 0]), 1))  # Initialize Dummy Array

    for chan_index, channel in enumerate(Features):  # Over all Channels
        for freq_index, frequency in enumerate(channel):  # For Range of All Frequency Bins
            ordered_trials = np.concatenate((ordered_trials, frequency), axis=1)
            for temps in range(len(frequency[0, :])):
                # TODO: Refactor pearson_ml_module to run faster
                universal_index = (chan_index, freq_index, temps)  # Tuple contains (Channel #, Freq Band #)
                column_index.append(universal_index)  # Append Index Tuple in Column Order
    ordered_trials = np.delete(ordered_trials, 0, 1)  # Delete the First Row (Initialized Row)
    return ordered_trials, column_index


def ml_order_pearson(Extracted_Features):
    """Restructures Pearson Corr. Coeff. Features to a ndarray structure usable to SciKit-Learn: (n_samples, n_features)

    Parameters:
    -----------
    Extracted_Features:l ist
        list of Pearson Correlation Values between each instance and the LFP Template of each Label
        [Labels]->[ch] -> [freq] -> ( Number of Instances  x  Number of Labels/Templates)

    Returns:
    --------
    ML_Ready: ndarray
        Array that is structured to work with the SciKit-learn Package
        (n_samples, n_features)
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq * Num_Temps)
    ML_Labels: ndarray
        1-d array of Labels of the Corresponding n_samples
        ( n_samples   x   1 )
    Ordered_Index: list
        Index of Features for Feature Dropping
        [Num of Features] -> (Chan Num , Freq Num, Temp Num)
                     list -> Tuple

    """
    ml_ready = np.zeros((1, (len(Extracted_Features[0]) * len(Extracted_Features[0][0]) * len(Extracted_Features))))
    ml_labels = np.zeros((1, 1))
    for label_index, label in enumerate(Extracted_Features):
        ordered_trials, ordered_index = pearson_ml_module(label)
        ml_ready = np.concatenate((ml_ready, ordered_trials), axis=0)

        # Handles Labels so they are flexible when grouping
        row, coll = np.shape(ordered_trials)
        dyn_labels = np.zeros([row, 1])
        dyn_labels[:, 0] = label_index
        ml_labels = np.concatenate((ml_labels, dyn_labels), axis=0)

    ml_ready = np.delete(ml_ready, 0, 0)
    ml_labels = np.delete(ml_labels, 0, 0)
    return ml_ready, ml_labels, ordered_index


def Power_Extraction(Clipped_Trials):
    """Finds the Power for all Labels for each Channels Frequency Band

    Parameters:
    -----------
    Clipped_Trials: list
        List containing the Segments Clipped by Label_Extract_Pipeline
        [labels] -> [ch] -> [freq] -> ( Samples x Label Instances)

    Return:
    -------
    extracted_power: list
        List of power for each Channel for every Frequency Band for every instances for Each Label
        [labels] -> [ch] -> [freq] -> (Mean of Samples for each Label Instances   x   1)
                                        (Num Label Instances x 1)
    """
    extracted_power = []
    for label in Clipped_Trials:
        extracted_power.append(find_power(label))
    return extracted_power


def ml_order_power(Extracted_Features):
    """Restructure Extracted Power Features to a ndarray structure usable to SciKit-Learn: (n_samples, n_features)

    Information:
    ------------
        ml is short for Machine Learning. Read: Machine Learning Order Power

    Parameters:
    -----------
    Extracted_Features:list
        List of power for each Channel for every Frequency Band for every instances for Each Label
        [labels] -> [ch] -> [freq] -> (Mean of Samples for each Label Instances   x   1)
                                        (Num Label Instances x 1)

    Returns:
    --------
    ML_Ready: ndarray
        Array that is structured to work with the SciKit-learn Package
        (n_samples, n_features)
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    ML_Labels: ndarray
        1-d array of Labels of the Corresponding n_samples
        ( n_samples   x   1 )
    Ordered_Index: list
        Index of Features for Feature Dropping
        [Num of Features] -> (Chan Num , Freq Num)
        list -> Tuple

    """

    ml_ready = np.zeros((1, (len(Extracted_Features[0]) * len(Extracted_Features[0][0]))))
    ml_labels = np.zeros((1,))
    for label in range(len(Extracted_Features)):
        ordered_trials, Ordered_Index = power_ml_order_module(Extracted_Features[label])
        ml_ready = np.concatenate((ml_ready, ordered_trials), axis=0)

        # Handles Labels so they are flexible when grouping
        row, coll = np.shape(ordered_trials)
        dyn_labels = np.zeros([row, ])
        dyn_labels[:] = label
        ml_labels = np.concatenate((ml_labels, dyn_labels), axis=0)

    ml_ready = np.delete(ml_ready, 0, 0)
    ml_labels = np.delete(ml_labels, 0, 0)
    return ml_ready, ml_labels, Ordered_Index


def Select_Classifier(Model: str, Strategy: str):
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

def power_ml_order_module(Features):
    """Reorganizes the Extracted Power Features of One Lable's instance/samples into a Useful Machine Learning Format

    Parameters:
    -----------
    Features: list
        List of power for each Channel for every Frequency Band for every instances for One Label
        [ch] -> [freq] -> (Mean of Samples for each Label Instances   x   1)
                                        (Num Label Instances x 1)
    Returns:
    --------
    Ordered_Trials: ndarray
        Array of All Instance/Sample's Features for the Label Passes
        (1 x Num Features)     * Note: Num Features = Num Chans x Num Freq
    Column_Index: list
        Index of Features for Feature Dropping
            [Num of Features] -> (Chan Num , Freq Num)     * Note: list -> Tuple
    """
    # Create Variable for Indexing
    B = len(Features[0][0][:, 0])  # Length of Dynam. Clipped Training Set

    # Create Initial Array
    Column_Index = []
    Ordered_Trials = np.zeros((B, 1))  # Initialize Dummy Array

    # Channel Based Ordering
    for chan_index, channel in enumerate(Features):  # Over all Channels
        for freq_index, frequency in enumerate(channel):  # For Range of All Frequency Bins
            Ordered_Trials = np.concatenate((Ordered_Trials, frequency), axis=1)
            Column_Index.append((chan_index, freq_index))  # Append Index Tuple in Column Order
    Ordered_Trials = np.delete(Ordered_Trials, 0, 1)  # Delete the First Row (Initialized Row)

    return Ordered_Trials, Column_Index


def Slider(Ext_Starts, full_trial, Slide: int, Step=False):
    """ Slider is a parameter to increase the number of samples around the onset of a behavior of interest
    Parameters:
    -----------
    Ext_Starts: list
        List of all Start Times corresponding to ONE Label in each Epoch in Full_Trials
        [Epochs]->[Start Time]
    full_trial: int

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
                    if (Ext_Starts[i][j] + k) <= full_trial:
                        Slid_Trial.append(Ext_Starts[i][j] + k)
            if Step == True:
                for k in range(0, Slide, Step):
                    if (Ext_Starts[i][j] + k) <= full_trial:
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
                         verbose=False, n_folds=4):  # , SKF, verbose=False):
    """ This Function is a Flexible Machine Learning Function that Trains FOUR Classifiers and determines metrics of each
    The metrics it determines are:
                [1] Accuracy & StdERR
                [2] Confusion Matrix
                [3] Reciever Operating Curve
    It stores all of the trained Classifiers into a Dictionary and pairs it with a
    Dictionary of the Corresponding Test Index. The Pair are returned as a tuple


    Parameters:
    -----------
    n_folds: int
        number of folds to use to cross-validate


    Returns:
    --------



    """

    num_k_folds = n_folds
    class_index = list(set(Data_Labels))
    # X = Data_Set
    # y = Data_Labels

    #     y = label_binarize(Data_Labels, classes=list(set(Data_Labels)))

    skf = StratifiedKFold(Data_Labels, n_folds=num_k_folds)

    acc = np.zeros(num_k_folds)
    all_conf_mats = []  # Just Added
    fold_num = 0

    trained_classifiers = dict()
    trained_index = dict()
    for train, test in skf:
        if verbose:
            print("Fold %s..." % fold_num)
        # print "%s %s" % (train, test)
        data_train, data_train = Data_Set[train, :], Data_Labels[train]
        data_test, data_test = Data_Set[test, :], Data_Labels[test]

        classifier = Select_Classifier(Model=Method, Strategy=Strategy)
        classifier.fit(data_train, data_train)

        y_pred = classifier.predict(data_test)

        confusion = confusion_matrix(data_test, y_pred).astype(float)
        num_test_trials = len(data_test)
        acc[fold_num] = sum(np.diag(confusion)) / num_test_trials

        trained_classifiers[fold_num] = classifier
        trained_index[fold_num] = test
        fold_num += 1

        if verbose:
            print(confusion)
        all_conf_mats.append(confusion)

    mean_acc_nb = np.mean(acc)
    std_err_nb = np.std(acc) / np.sqrt(num_k_folds)
    classifier_components = (trained_classifiers, trained_index)

    if verbose:
        print("cross-validated acc: %.2f +/- %.2f" % (np.mean(acc), np.std(acc)))

    return mean_acc_nb, std_err_nb, classifier_components, all_conf_mats


def Make_Full_Trial_Index(Features, Offset: int, Tr_Length: int, Epoch_Len: int):
    """

    Parameters:
    -----------
    Features: list
        Features: list
        [Ch]->[Freq]->(Time Samples x Trials)
    Offset: int
        How much prior or after onset
    Tr_Length: int
        How many samples to use for window
    Epoch_Len: int
        The length in Samples of each Epoch
    Returns:
    --------
    FT_Index =

    """

    FT_Index = []
    for i in range(len(Features[0][0][0, :])):
        if Offset + Tr_Length >= 0:
            time_list = np.arange(Offset + Tr_Length, Epoch_Len)
        else:  # Prevents unintended indexing if neural activity is include non-causal information
            time_list = np.arange(0, Epoch_Len + Offset + Tr_Length)
        FT_Index.append(list(time_list))
    return FT_Index


# Function needs to be able to Selectively choose which Trials to run full Trial
def Series_LFP_Clipper(Features, Offset: int, Tr_Length: int):
    """This Function Sequentially clips Neural data attributed for a full trial and organizes
    them for future steps. It iterates over EACH full trial clipping.
    It should be run once.

    Its output can later be broken into an Initial Training and Final Test Set.


    Parameters:
    -----------
    Features: list
        [Ch]->[Freq]->(Time Samples x Epochs)
    Offset: int
        How much prior or after onset
    Tr_Length: int
        How many samples to use for window

    Returns:
    --------
    dynamic_freq_trials: list
        [ch]->[freq]->(Samples x Trials/Observations)

    """

    starts = Make_Full_Trial_Index(Features, Offset=Offset, Tr_Length=Tr_Length, Epoch_Len=len(Features[0][0][:, 0]))

    nt = len(Features[0][0][0, :])  # Number of Trials of Dynam. Clipped Training Set
    n_el = Numel(starts) - Numbad(starts, Offset=Offset, Tr_Length=Tr_Length)  # Number of Examples

    dynamic_freq_trials = []
    for channel in Features:  # Over all Channels
        freq_trials = []
        for frequency in channel:  # For Range of All Frequency Bins
            chan_holder = np.zeros((Tr_Length, n_el))  # Initiate Holder for Trials (Motifs)
            counter = 0  # For stackin all examples of label in full trial
            for trials in range(nt):
                for ex in range(len(starts[trials])):
                    if starts[trials][ex] - Offset - Tr_Length >= 0 and starts[trials][ex] - Offset <= len(
                            Features[0][0][:, 0]):
                        chan_holder[:, counter] = frequency[
                                                  starts[trials][ex] - Offset - Tr_Length:starts[trials][ex] - Offset,
                                                  trials]  # Select Motif
                        counter = counter + 1
            freq_trials.append(chan_holder)  # Save all of the Trials for that Frequency on that Channel
        dynamic_freq_trials.append(freq_trials)  # Save all of the Trials for all Frequencies on each Channel

    return dynamic_freq_trials


def series_label_extractor(labels, clippings, label_instructions, Offset: int, Tr_Length: int, undetermined=False):
    """Creates a list of each epoch's series labels to be paired with the lfp data for Machine Learning

    :param labels:
    :param clippings:
    Label_Instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Offset = int
        The number of samples away from the true onset to Grab for ML Trials (Can be Before or After)
    Tr_Length=int
        Number of Samples to use for Features
    :param undetermined:
    Returns:
    --------
    series_labels: list
        [epoch]->(Epoch length in Samples - Offset - Tr_Length x 1)
    """

    series_labels = []
    for epoch in range(len(labels)):
        epoch_labels = Create_Label_Timeline(labels=labels,
                                             clippings=clippings,
                                             sel_epoch=epoch,
                                             label_instructions=label_instructions,
                                             undetermined=undetermined)
        if Offset + Tr_Length >= 0:
            series_labels.append(epoch_labels[Offset + Tr_Length:, 0])
        else:  # Prevents unintended indexing if neural activity is include non-causal information
            series_labels.append(epoch_labels[:Offset + Tr_Length, 0])
    return series_labels


def label_conversion(label, instructions, spec_instr):
    """Function converts the labels to a integer that is of the structure of the label_instructions parameter

        Parameters:
        -----------
        :param spec_instr:
        :param label:
        Label_Instructions: list
            list of labels and how they should be treated. If you use a nested list in the instructions the labels in
            this nested list will be treated as if they are the same label

        Returns:
        --------
        conversion: int
            Machine Learning encoding of labels based on label_instructions
        """
    count = 0
    # for instruction in instructions:
    if label in instructions:
        conversion = instructions.index(label)
    else:
        for instruction in instructions:
            if isinstance(instruction, list):
                if label in instruction:
                    conversion = int(count)
            elif isinstance(spec_instr, int):
                conversion = spec_instr
            else:
                print(" You did not include one of the labels in your instructions (Likely 'C')")
                return
            count += 1
    return conversion


def Create_Label_Timeline(labels, clippings, sel_epoch, label_instructions, undetermined=False):
    """ Creates a timeline of the syllable Labels for Visualization

    Parameters:
    -----------
    labels:

    clippings:

    sel_epoch:


    Label_Instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label


    Returns:
    --------
    conversion: int
        Machine Learning encoding of labels based on label_instructions

    """

    time_series_labels = []
    # # Make Dummy List (THIS is really bad coding)
    # for i in range(4500):
    #     timeseries_labels.append(6)  # 6 is Silence for original Labling
    time_series_labels = np.zeros((4500, 1))

    #     TIMESERIES_LABELS = np.zeros([4500, 1])

    for labels, starts, ends, in zip(labels[sel_epoch], clippings[0][sel_epoch], clippings[1][sel_epoch]):
        sel_label = labels
        start_int = int(starts / 30)
        end_int = int(ends / 30)
        time_series_labels[start_int: end_int, 0] = label_conversion(sel_label,
                                                                     label_instructions,
                                                                     spec_instr=undetermined)
        # for internal_ind in range(end_int - start_int):
        #     if (start_int + internal_ind) < 4500:
        #         time_series_labels[start_int + internal_ind, 0] = label_conversion(sel_label, label_instructions)

    return time_series_labels.astype(int)


# Clip_Test, Temp_Test = Label_Extract_Pipeline(Dataset,  Stereotype_Labels, Stereotype_Clippings[0],  [1,2,3,4], Offset = 10, Tr_Length= 20)
# Power_List = Power_Extraction(Clip_Test)
# ML_Trial_Test, ML_Labels_Test, Ordered_Index_TEST = ml_order_power(Power_List)


def Classification_Prep_Pipeline(Full_Trials, All_Labels, Time_Stamps, Label_Instructions, Offset: int, Tr_Length: int,
                                 Feature_Type: str, Temps=None, Slide=None, Step=False):
    """

    Parameters:
    -----------
    :param Offset:
    :param Tr_Length:
    :param Full_Trials:
    :param All_Labels:
    :param Time_Stamps:
    Label_Instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Offset = int
        The number of samples away from the true onset to Grab for ML Trials (Can be Before or After)
    Tr_Length=int
        Number of Samples to use for Features
    Feature_Type: str
        Options [Power, Pearson]
        TODO: Populate the Docustring of the CLassification_Prep_Pipeline
    :param Temps:
    :param Slide:
    :param Step:

    Returns:
    --------
    ML_Trials: ndarray
        Array that is structured to work with the SciKit-learn Package
        (n_samples, n_features)
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    ML_Labels: ndarray
        1-d array of Labels of the Corresponding n_samples
        ( n_samples   x   1 )
    Ordered_Index: list
        Index of Features for Feature Dropping
                            [list] -> (Tuple)
        Power:   [Num of Features] -> (Chan Num , Freq Num)
        Pearson: [Num of Features] -> (Chan Num , Freq Num, Temp Num)
    """

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
        ML_Trials, ML_Labels, Ordered_Index = ml_order_power(Power)

    if Feature_Type == 'Pearson':
        print('Probably need to Validate Pearson Functions Correctly')
        # Function for Finding Pearson
        ## [Probably should add *kwargs]
        # Fucntion for Ordering Pearson
        if Temps is None:
            pearson = pearson_extraction(Clips, Temps_internal)
        elif Temps is not None:
            pearson = pearson_extraction(Clips, Temps)

        ML_Trials, ML_Labels, Ordered_Index = ml_order_pearson(pearson)

        if Temps is None:
            return ML_Trials, ML_Labels, Ordered_Index, Temps_internal

    return ML_Trials, ML_Labels, Ordered_Index


def Series_Classification_Prep_Pipeline(Features, Offset: int, Tr_Length: int, Feature_Type: str, labels: list,
                                        onsets: list, label_instructions: list, Temps=None, re_break=False):
    """

    Parameters:
    -----------
    Features: list
        [Ch]->[Freq]->(Time Samples x Trials)
    Offset: int

    Tr_Length: int

    Feature_Type: int

    Temps:

    re_break: bool (Optional)
        Option for Breaking up the epoch for visualizing their individual Testing performances (Defaults: False)

    Returns:
    --------
    #TODO: rename variable and document
    series_ordered: list
        list of ndarrays of All Instance/Sample's Features for the Label Passes
        [Epoch]->[Samples/Time x Features]
                    (1 x Num Features)     * Note: Num Features = Num Chans x Num Freq
    #TODO: Improve Documentation
    #TODO: Determine the shape of the series_ordered output use the ML_Trials info found below
    # ML_Trials: ndarray
    #     Array that is structured to work with the SciKit-learn Package
    #     (n_samples, n_features)
    #         n_samples = Num of Instances Total
    #         n_features = Num_Ch * Num_Freq)
    ml_labels: ndarray
        1-d array of Labels of the Corresponding n_samples
        ( n_samples   x   1 )

    ordered_index: list
        Index of Features for Feature Dropping
                            [list] -> (Tuple)
        Power:   [Num of Features] -> (Chan Num , Freq Num)
        Pearson: [Num of Features] -> (Chan Num , Freq Num, Temp Num)

    """

    Series_Trial = Series_LFP_Clipper(Features, Offset=Offset, Tr_Length=Tr_Length)

    if Feature_Type == 'Power':
        series_power = find_power(Series_Trial)
        series_ordered, ordered_index = power_ml_order_module(series_power)

    elif Feature_Type == 'Pearson':
        series_pearson = find_pearson_coeff(Series_Trial, Temps)
        series_ordered, ordered_index = pearson_ml_module(series_pearson)

    elif Feature_Type == 'Both':
        # Handle Power First
        series_power = find_power(Series_Trial)
        series_ordered_power, ordered_index_power = power_ml_order_module(series_power)
        # Handle Pearson Coefficient Second
        series_pearson = find_pearson_coeff(Series_Trial, Temps)
        series_ordered_pearson, ordered_index_pearson = pearson_ml_module(series_pearson)
        # Concatenate their lists
        series_ordered = np.concatenate((series_ordered_power, series_ordered_pearson), axis=1)
        ordered_index = np.concatenate((ordered_index_power, ordered_index_pearson), axis=0)


    else:
        print(" You didn't input a Valid Feature Type")
        return

    # Convert Labels to Scikit-Learn format
    ml_labels = series_ml_order_label(series_label_extractor(labels, onsets, label_instructions,
                                                             Offset=Offset, Tr_Length=Tr_Length))

    if re_break == True:  # Option for Re-Breaking the epoch to get visualize their individual performances in Testing
        # Break the long time series back into the Constituent Epochs
        full_trial_features = []
        full_trial_labels = []
        trial_length = len(Features[0][0][:, 0]) - abs(Offset + Tr_Length)
        for i in range(len(Features[0][0][0, :])):
            full_trial_features.append(series_ordered[trial_length * i:trial_length * (i + 1), :])
            full_trial_labels.append(ml_labels[trial_length * i:trial_length * (i + 1)])
        return full_trial_features, full_trial_labels, ordered_index

    return series_ordered, ml_labels, ordered_index


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
    """ This Function is a Flexible Machine Learning Function that Trains One Classifier and determines metrics for it
    The metrics it determines are:
                [1] Accuracy & StdERR
                [2] Confusion Matrix
                [3] Reciever Operating Curve
    It stores all of the trained Classifiers into a Dictionary and pairs it with a
    Dictionary of the Corresponding Test Index. The Pair are returned as a tuple

    Parameters:
    -----------
    Class_Obj: class
        classifier object from the scikit-learn package
    Train_Set:

    Train_LabelsL

    Test_Set:

    Test_Labels:

    verbose: bool
        If True it prints useful characteristics of the trained model, defaults to False

    Returns:
    --------
    acc: int
        the accuracy of the trained classifier
    classifier: class
        a trained classifier dictacted by the CLass_Object Parameter from scikit-learn
    confusion: array
        Confusion matrix, shape = [n_classes, n_classes]

    """

    classifier = Class_Obj
    classifier.fit(Train_Set, Train_Labels)

    test_pred = classifier.predict(Test_Set)

    confusion = confusion_matrix(Test_Labels, test_pred).astype(float)
    numTestTrials = len(Test_Labels)

    acc = sum(np.diag(confusion)) / numTestTrials
    return acc, classifier, confusion


def Trial_Selector(Features, Sel_index):
    """This Function allows you to easily parse our specific Trials for K-Fold validation
    and Test Set Seperation

    """
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


def Label_Selector(labels, sel_index):
    """This Function allows you to easily parse out specific Trial's Labels for K-Fold validation
    and Test Set Seperation

    Parameters:
    ----------
    labels:

    sel_index:

    Returns:
    --------
    sel_labels:

    """

    sel_labels = []
    for i in range(len(sel_index)):
        sel_labels.append(labels[sel_index[i]])
    return sel_labels


def Convienient_Selector(Features, Labels, Starts, Sel_index):
    """Abstractly reorganizes the list of Epochs and Labels to ndarray compatible with scikitlearn

    :param Features:
    :param Labels:
    :param Starts:
    :param Sel_index:
    :return:

    # sel_set: list
    #     list of the selected K-Fold's Training set
    #     [ch] -> [Freq] -> (Time x Num_Epoxhs)
    # sel_labels: list
    #     list of each Epoch/Samples labels
    #     [Trial/EPoch] -> [labels]
    # (sel_starts, sel_ends): tuple
    #     Tuple of the Label Onsets
    #         ( [Stars] , [Ends] )
    """

    sel_set = Trial_Selector(Features=Features, Sel_index=Sel_index)
    sel_labels = Label_Selector(Labels, sel_index=Sel_index)
    sel_starts = Label_Selector(Starts, sel_index=Sel_index)
    return sel_set, sel_labels, sel_starts

    # def Series_Convienient_Selector(Features, Labels, Onsets, Sel_index):
    #     """Abstractly reorganizes the list of Epochs and Labels to ndarray compatible with scikitlearn
    #
    #     :param Onsets:
    #     :param Features:
    #     :param Labels:
    #     :param Sel_index:
    #
    #     Returns:
    #     sel_set: list
    #         list of the selected K-Fold's Training set
    #         [ch] -> [Freq] -> (Time x Num_Epoxhs)
    #     sel_labels: list
    #         list of each Epoch/Samples labels
    #         [Trial/EPoch] -> [labels]
    #     (sel_starts, sel_ends): tuple
    #         Tuple of the Label Onsets
    #             ( [Stars] , [Ends] )
    #     """
    # starts = Onsets[0]
    # ends = Onsets[1]
    # sel_set = Trial_Selector(Features=Features, Sel_index=Sel_index)
    # sel_labels = Label_Selector(Labels, sel_index=Sel_index)
    # sel_starts = Label_Selector(starts, sel_index=Sel_index)
    # sel_ends = Label_Selector(ends, sel_index=Sel_index)
    # return sel_set, sel_labels, (sel_starts, sel_ends)


#### NEED TO AD OPTIONAL IF STATETMENT HANDLING FOR RETURNING THE TEMPLATES FOR SERIES CLASSIFICATION

def clip_kfold(Class_Obj, Data_Set, Data_Labels, Data_Starts, Label_Instructions, Offset=int, Tr_Length=int,
               Feature_Type=str, k_folds=4, Slide=None, Step=False, verbose=False):
    """

    Class_Obj: class
        classifier object from the scikit-learn package
    :param Data_Set:
    Data_Labels: list
        List of all Labels corresponding to each Epoch in Full_Trials
        [Epochs]->[Labels]
    Data_Starts: list
        List of all Start Times corresponding to each Epoch in Full_Trials
        [Epochs]->[Start Time]
    Label_Instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Offset = int
        The number of samples away from the true onset to Grab for ML Trials (Can be Before or After)
    Tr_Length=int
        Number of Samples to use for Features
    Feature_Type: str
        Options: ['Power','Pearson']
    k_folds: int
        Number of Folds for Cross-Validation
    Slide: bool (Optional)
        #TODO: Invesitage Slide Parameter in clipkfold
    Step:
        #TODO: Investigate and Document the Step Parameter
    verbose: bool
        If True the function will print out messages to update user on its progress

    Returns:
    --------
    mean_acc_nb: int
        the mean accuracy across the folds
    std_err_nb: int
        the standard error across the folds
    classifier_components: tuple
        Tuples containing two Dictionaries with the fold number being the keys (using 0 indexing).
        Their Values are:
            1.) trained Classifier instances and the index of features for their models
                The values are that fold's trained classifier instance of the the CLass_Object Parameter
                (from scikit-learn)
            2.) list of the Test set for the corresponding trained Classifier
       shape =  ({ Fold_Num: Trained_Classifiers }, {Fold_Num: Test_Index})
    confusion: list
        list of each fold's Confusion matrix, shape = [n_classes, n_classes]

    """
    #     Data_Set, Data_Labels, Data_Starts, Label_Instructions, Offset = int, Tr_Length= int, Feature_Type = str) , Temps = None

    skf = StratifiedKFold(n_splits=k_folds)

    acc = np.zeros(k_folds)
    confusion = []  # Just Added
    # ROC = []  # Just Added too 8/10
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

        if Feature_Type == 'Power':
            ml_train_trials, ml_train_labels, train_ordered_index = Classification_Prep_Pipeline(train_set,
                                                                                                 train_labels,
                                                                                                 train_starts,
                                                                                                 Label_Instructions,
                                                                                                 Offset=Offset,
                                                                                                 Tr_Length=Tr_Length,
                                                                                                 Feature_Type=Feature_Type,
                                                                                                 Temps=None,
                                                                                                 Slide=Slide,
                                                                                                 Step=Step)
            ml_test_trials, ml_test_labels, test_ordered_index = Classification_Prep_Pipeline(test_set,
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
            ml_train_trials, ml_train_labels, train_ordered_index, temps_int = Classification_Prep_Pipeline(train_set,
                                                                                                            train_labels,
                                                                                                            train_starts,
                                                                                                            Label_Instructions,
                                                                                                            Offset=Offset,
                                                                                                            Tr_Length=Tr_Length,
                                                                                                            Feature_Type=Feature_Type,
                                                                                                            Temps=None,
                                                                                                            Slide=Slide,
                                                                                                            Step=Step)

            ml_test_trials, ml_test_labels, test_ordered_index = Classification_Prep_Pipeline(test_set,
                                                                                              test_labels,
                                                                                              test_starts,
                                                                                              Label_Instructions,
                                                                                              Offset=Offset,
                                                                                              Tr_Length=Tr_Length,
                                                                                              Feature_Type=Feature_Type,
                                                                                              Temps=temps_int,
                                                                                              Slide=Slide,
                                                                                              Step=Step)

        acc[foldNum], Trained_Classifiers[foldNum], conf = Clip_Classification(Class_Obj, ml_train_trials,
                                                                               ml_train_labels,
                                                                               ml_test_trials, ml_test_labels,
                                                                               verbose=False)
        Trained_Index[foldNum] = test
        foldNum += 1

        if verbose:
            print(conf)
        confusion.append(conf)

    meanacc_nb = np.mean(acc)
    stderr_nb = np.std(acc) / np.sqrt(k_folds)
    classifier_components = (Trained_Classifiers, Trained_Index)

    if verbose:
        print("cross-validated acc: %.2f +/- %.2f" % (np.mean(acc), np.std(acc)))
    return meanacc_nb, stderr_nb, classifier_components, confusion,


# TODO: FInisht the Train_on_All Function. It is unoperatable and incomplete
# def train_on_all(Class_Obj, Data_Set, Data_Labels, Data_Starts, Label_Instructions, Offset=int, Tr_Length=int,
#                Feature_Type=str, Slide=None, Step=False, verbose=False):
#     """
#
#     :param Class_Obj:
#     :param Data_Set:
#     :param Data_Labels:
#     :param Data_Starts:
#     Label_Instructions: list
#         list of labels and how they should be treated. If you use a nested list in the instructions the labels in
#         this nested list will be treated as if they are the same label
#     Offset = int
#         The number of samples away from the true onset to Grab for ML Trials (Can be Before or After)
#     Tr_Length=int
#         Number of Samples to use for Features
#     :param Feature_Type:
#     :param k_folds:
#     :param Slide:
#     :param Step:
#     :param verbose:
#
#     Returns:
#     --------
#
#
#     """
#     #     Data_Set, Data_Labels, Data_Starts, Label_Instructions, Offset = int, Tr_Length= int, Feature_Type = str) , Temps = None
#
#     # skf = StratifiedKFold(n_splits=k_folds)
#
#     # acc = np.zeros(k_folds)
#     # confusion = []  # Just Added
#     # ROC = []  # Just Added too 8/10
#     # foldNum = 0
#
#     Trained_Classifiers = dict()
#     Trained_Index = dict()
#
#     Num_Clippings = np.ones(len(Data_Labels))
#
#     # for train, test in skf.split(Num_Clippings, Num_Clippings):
#         if verbose:
#             print("Fold %s..." % foldNum)
#             # print "%s %s" % (train, test)
#
#         # print(train)
#         # train_set, train_labels, train_starts = Convienient_Selector(Data_Set, Data_Labels, Data_Starts, )
#
#         # print(test)
#         # test_set, test_labels, test_starts = Convienient_Selector(Data_Set, Data_Labels, Data_Starts, test)
#
#         # if Feature_Type != 'Pearson':
#         ml_train_trials, ml_train_labels, train_ordered_index = Classification_Prep_Pipeline(Data_Set,
#                                                                                              Data_Labels,
#                                                                                              Data_Starts
#                                                                                              Label_Instructions,
#                                                                                              Offset=Offset,
#                                                                                              Tr_Length=Tr_Length,
#                                                                                              Feature_Type=Feature_Type,
#                                                                                              Temps=None,
#                                                                                              Slide=Slide,
#                                                                                              Step=Step)
#
#         # ml_test_trials, ml_test_labels, test_ordered_index = Classification_Prep_Pipeline(test_set,
#         #                                                                                   test_labels,
#         #                                                                                   test_starts,
#         #                                                                                   Label_Instructions,
#         #                                                                                   Offset=Offset,
#         #                                                                                   Tr_Length=Tr_Length,
#         #                                                                                   Feature_Type=Feature_Type,
#         #                                                                                   Temps=None,
#         #                                                                                   Slide=Slide,
#         #                                                                                   Step=Step)
#
#         if Feature_Type == 'Pearson':
#             ml_train_trials, ml_train_labels, train_ordered_index, Temps_int = Classification_Prep_Pipeline(Data_Set,
#                                                                                                             Data_Labels,
#                                                                                                             Data_Starts
#                                                                                                             Offset=Offset,
#                                                                                                             Tr_Length=Tr_Length,
#                                                                                                             Feature_Type=Feature_Type,
#                                                                                                             Temps=None,
#                                                                                                             Slide=Slide,
#                                                                                                             Step=Step)
#
#             ml_test_trials, ml_test_labels, test_ordered_index = Classification_Prep_Pipeline(test_set,
#                                                                                               test_labels,
#                                                                                               test_starts,
#                                                                                               Label_Instructions,
#                                                                                               Offset=Offset,
#                                                                                               Tr_Length=Tr_Length,
#                                                                                               Feature_Type=Feature_Type,
#                                                                                               Temps=Temps_int,
#                                                                                               Slide=Slide,
#                                                                                               Step=Step)
#
#         acc[foldNum], Trained_Classifiers[foldNum], conf = Clip_Classification(Class_Obj, ml_train_trials, ml_train_labels,
#                                                                             ml_test_trials, ml_test_labels,
#                                                                             verbose=False)
#         Trained_Index[foldNum] = test
#         foldNum += 1
#
#         if verbose:
#             print(conf)
#         confusion.append(conf)
#
#     meanAcc_nb = np.mean(acc)
#     stdErr_nb = np.std(acc) / np.sqrt(k_folds)
#     Classifier_Components = (Trained_Classifiers, Trained_Index)
#
#     if verbose:
#         print("cross-validated acc: %.2f +/- %.2f" % (np.mean(acc), np.std(acc)))
#     return meanAcc_nb, stdErr_nb, Classifier_Components, confusion,

###
# Series Analysis Code


def series_ml_order_label(labels: list):
    """ Convert labels to a format that will work with Scikit-Learn
    ParametersL
    labels: int
        [epoch]->[labels]
    return:
    ml_labels: ndarray
        restructure labels to work nicely with 
    """

    ml_labels = np.zeros((1,))
    for epoch in labels:
        ml_labels = np.concatenate((ml_labels, epoch), axis=0)
    ml_labels = np.delete(ml_labels, 0, 0)

    return ml_labels


# def KFold_Series_Prep(Data_Set, Test_index, Offset=int, Tr_Length=int, Feature_Type=str):
#     """ Handles the Preparation for series_clip_kfold
#
#     :param Data_Set:
#     :param Test_index:
#     :param Offset:
#     :param Tr_Length:
#     :param Feature_Type:
#     :return:
#     """
#     Trial_set = Trial_Selector(Features=Data_Set, Sel_index=Test_index)
#
#     series_ready = Series_Classification_Prep_Pipeline(Trial_set, Offset=Offset, Tr_Length=Tr_Length,
#                                                        Feature_Type=Feature_Type, Temps=temps)
#
#     return series_ready,


def Series_Convienient_Selector(Features, Labels, Onsets, Sel_index):
    """Abstractly reorganizes the list of Epochs and Labels to ndarray compatible with scikitlearn

    :param Onsets:
    :param Features:
    :param Labels:
    :param Sel_index:

    Returns:
    sel_set: list
        list of the selected K-Fold's Training set
        [ch] -> [Freq] -> (Time x Num_Epoxhs)
    sel_labels: list
        list of each Epoch/Samples labels
        [Trial/EPoch] -> [labels]
    (sel_starts, sel_ends): tuple
        Tuple of the Label Onsets
            ( [Stars] , [Ends] )
    """

    starts = Onsets[0]
    ends = Onsets[1]
    sel_set = Trial_Selector(Features=Features, Sel_index=Sel_index)
    sel_labels = Label_Selector(Labels, sel_index=Sel_index)
    sel_starts = Label_Selector(starts, sel_index=Sel_index)
    sel_ends = Label_Selector(ends, sel_index=Sel_index)
    return sel_set, sel_labels, (sel_starts, sel_ends)


# TODO: Add Parameter that will allow for pior jutter inclusion of a behavior to just prior its the True Onset
def series_clip_kFold(Class_Obj, Data_Set, Data_Labels, Data_Onsets, Label_Instructions, Offset=int, Tr_Length=int,
                      Feature_Type=str, k_folds=4, verbose=False):
    """
    Parameters:
    -----------
    Class_Obj: class
        classifier object from the scikit-learn package
    :param Data_Set:
    Data_Labels: list
        List of all Labels corresponding to each Epoch in Full_Trials
        [Epochs]->[Labels]
    Data_Starts: list
        List of all Start Times corresponding to each Epoch in Full_Trials
        [Epochs]->[Start Time]
    Label_Instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Offset = int
        The number of samples away from the true onset to Grab for ML Trials (Can be Before or After)
    Tr_Length=int
        Number of Samples to use for Features
    Feature_Type: str
        Options: [Power','Pearson']
    k_folds: int
        Number of Folds for Cross-Validation
    Slide: bool (Optional)
        #TODO: Invesitage Slide Parameter in clipkfold
    verbose: bool
        If True the function will print out messages to update user on its progress

    Returns:
    --------
    mean_acc_nb: int
        the mean accuracy across the folds
    std_err_nb: int
        the standard error across the folds
    classifier_components: tuple
        Tuples containing two Dictionaries with the fold number being the keys (using 0 indexing).
        Their Values are:
            1.) trained Classifier instances and the index of features for their models
                The values are that fold's trained classifier instance of the the CLass_Object Parameter
                (from scikit-learn)
            2.) list of the Test set for the corresponding trained Classifier
       shape =  ({ Fold_Num: Trained_Classifiers }, {Fold_Num: Test_Index})
    confusion: list
        list of each fold's Confusion matrix, shape = [n_classes, n_classes]

    """
    #     Data_Set, Data_Labels, Data_Starts, Label_Instructions, Offset = int, Tr_Length= int, Feature_Type = str) , Temps = None

    skf = StratifiedKFold(n_splits=k_folds)

    acc = np.zeros(k_folds)
    confusion = []  # Just Added
    foldNum = 0

    Trained_Classifiers = dict()
    Trained_Index = dict()

    Num_Clippings = np.ones(len(Data_Labels))

    for train, test in skf.split(Num_Clippings, Num_Clippings):
        if verbose:
            print("Fold %s..." % foldNum)
            # print "%s %s" % (train, test)

        print(train)
        train_set, train_labels, train_onsets = Series_Convienient_Selector(Data_Set, Data_Labels, Data_Onsets,
                                                                            train)

        print(test)
        test_set, test_labels, test_onsets = Series_Convienient_Selector(Data_Set, Data_Labels, Data_Onsets, test)

        # Features, Offset = int, Tr_Length = int, Feature_Type = str, Temps = None

        if Feature_Type == 'Pearson' or Feature_Type == 'Both':
            _, templates = Label_Extract_Pipeline(Full_Trials=train_set, All_Labels=train_labels,
                                                  Starts=train_onsets[0],
                                                  Label_Instructions=Label_Instructions,
                                                  Offset=Offset,
                                                  Tr_Length=Tr_Length,
                                                  Slide=None,
                                                  Step=False)
        else:
            templates = None

        ml_train_trials, ml_train_labels, train_ordered_index = Series_Classification_Prep_Pipeline(Features=train_set,
                                                                                                    Offset=Offset,
                                                                                                    Tr_Length=Tr_Length,
                                                                                                    Feature_Type=Feature_Type,
                                                                                                    Temps=templates,
                                                                                                    labels=train_labels,
                                                                                                    onsets=train_onsets,
                                                                                                    label_instructions=Label_Instructions)

        ml_test_trials, ml_test_labels, test_ordered_index = Series_Classification_Prep_Pipeline(Features=test_set,
                                                                                                 Offset=Offset,
                                                                                                 Tr_Length=Tr_Length,
                                                                                                 Feature_Type=Feature_Type,
                                                                                                 Temps=templates,
                                                                                                 labels=test_labels,
                                                                                                 onsets=test_onsets,
                                                                                                 label_instructions=Label_Instructions)

        acc[foldNum], Trained_Classifiers[foldNum], conf = Clip_Classification(Class_Obj, ml_train_trials,
                                                                               ml_train_labels,
                                                                               ml_test_trials, ml_test_labels,
                                                                               verbose=False)
        Trained_Index[foldNum] = test
        foldNum += 1

        if verbose:
            print(conf)
        confusion.append(conf)

    mean_acc_nb = np.mean(acc)
    std_err_nb = np.std(acc) / np.sqrt(k_folds)
    classifier_components = (Trained_Classifiers, Trained_Index)

    if verbose:
        print("cross-validated acc: %.2f +/- %.2f" % (np.mean(acc), np.std(acc)))
    return mean_acc_nb, std_err_nb, classifier_components, confusion


########################################################################################################################
####################### Code for Visualizing the Characteristic of Trained Models ######################################
########################################################################################################################


# Functions For Plotting the Confusion Matrix

def plot_confusion_matrix(cm, Names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, Names, rotation=45)
    plt.yticks(tick_marks, Names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_Norm_confusion_matrix(cm, Names, verbose=False):
    """Normalize the confusion matrix by row (i.e by the number of samples in each class)"""

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if verbose == True:
        print('Normalized confusion matrix')
        print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, Names, title='Normalized confusion matrix')


#     plt.show()

def plot_all_confusion_matrix(cm, Names):
    f, ax = plt.subplots(2, 2, sharex='col', sharey='row')
    ax[1] = plot_Norm_confusion_matrix(cm[0], Names)
    ax[1].set_title('Sharing x per column, y per row')
    ax[2] = plot_Norm_confusion_matrix(cm[1], Names)
    ax[3] = plot_Norm_confusion_matrix(cm[2], Names)
    ax[4] = plot_Norm_confusion_matrix(cm[3], Names)
    plt.show()


def plot_mean_confusion_matrix(cm, Names):
    x, y = np.shape(cm[0])
    holder = np.zeros([x, y])
    plt.figure()
    for i in range(len(cm)):
        holder += cm[i]
    plot_Norm_confusion_matrix(holder, Names)
    plt.show()


# Visualize Classifier Performance Characteristics


def ROC_Indepth(y_test, y_score, n_classes, binarize=True):
    if binarize == True:
        Classes = np.arange(n_classes)
        y_test = label_binarize(y_test, classes=Classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def Plot_ROC_Indepth(y_test, y_score, n_classes, class_names, binarize=True, Name_This_Bool=False):
    fpr, tpr, roc_auc = ROC_Indepth(y_test, y_score, n_classes, binarize=binarize)

    lw = 2
    # Plot all ROC curves
    plt.figure()

    if Name_This_Bool == True:
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'tab:brown', 'tab:pink',
                    'tab:gray', 'tab:green', 'xkcd:dark olive green',
                    'xkcd:ugly yellow', 'xkcd:fire engine red', 'xkcd:radioactive green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


# Visualize Offline Series Performance

def Visualize_Psuedo_Real(Audio, Predictions, Offset=int, Tr_Len=int):
    """Creates Plot of how the Classifiers predictions look against the Behavior

    :param Audio:
    :param Predictions:
    :param Offset:
    :param Tr_Len:
    :return:
    """
    plt.figure(1, figsize=(20, 4))

    # First_Predictions_List = list(First_Predictions)
    colors = ['black', 'red', 'orange', 'yellow', 'pink', 'green', 'white']

    # for i in range(len(colors)):
    for index, predict in enumerate(Predictions):
        # if int(predict) == i:
        plt.axvline(x=(index + Offset + Tr_Len) * 30, color=colors[int(predict)])

    # This is a Hack Improve for Actual use:
    black_patch = mpatches.Patch(color='black', label='Syllable 1')
    red_patch = mpatches.Patch(color='red', label='Syllable 2')
    orange_patch = mpatches.Patch(color='orange', label='Syllable 3')
    yellow_patch = mpatches.Patch(color='yellow', label='Syllable 4')
    pink_patch = mpatches.Patch(color='pink', label='Introductory Note')
    green_patch = mpatches.Patch(color='green', label='Silence')
    white_patch = mpatches.Patch(color='white', label='Silence')

    #     plt.legend(handles=[red_patch])
    plt.legend(handles=[black_patch, red_patch, orange_patch, yellow_patch, pink_patch, green_patch, white_patch],
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.plot(Audio)


def Visualize_True_Audio_Labels(Audio, Predictions):
    """ Visualizes the True Label for Epoch Audio

    """
    plt.figure(1, figsize=(20, 4))

    Conversion_Dict = {1: 0, 2: 1, 3: 2, 4: 3, 'I': 4, 6: 5, 'C': 6}  # 6 is Silence for original Labling

    # First_Predictions_List = list(First_Predictions)
    colors = ['black', 'red', 'orange', 'yellow', 'pink', 'white']

    for i in range(len(colors)):
        for j in range(len(Predictions)):
            if Conversion_Dict[Predictions[j]] == i:
                plt.axvline(x=(j) * 30, color=colors[i])

    # This is a Hack Improve for Actual use:
    black_patch = mpatches.Patch(color='black', label='Syllable 1')
    red_patch = mpatches.Patch(color='red', label='Syllable 2')
    orange_patch = mpatches.Patch(color='orange', label='Syllable 3')
    yellow_patch = mpatches.Patch(color='yellow', label='Syllable 4')
    pink_patch = mpatches.Patch(color='pink', label='Introductory Note')
    white_patch = mpatches.Patch(color='white', label='Silence')

    #     plt.legend(handles=[red_patch])
    plt.legend(handles=[black_patch, red_patch, orange_patch, yellow_patch, pink_patch, white_patch],
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim(xmin=0, xmax=135000)
    plt.title('True Labels for Epoch')
    plt.plot(Audio)


# RODO: Determine what to do with this function
# TODO: Change the Performance visualization to be a fill between implementation istead of a vertical line
def series_performance_prep(Data_Set, Test_index, label_instructions, labels, onsets, Offset=int, Tr_Length=int,
                            Feature_Type=str):
    """Function grabs the true Labels for the Test of of Epoch and Returns them for Performance Visualizaiton

    :param Data_Set:
    :param Test_index:
    :param Offset:
    :param Tr_Length:
    :param Feature_Type:

    Returns:
    --------
    ml_trials

    ml_labels:

    ordered_index:
        Index of Features for Feature Dropping
                            [list] -> (Tuple)
        Power:   [Num of Features] -> (Chan Num , Freq Num)
        Pearson: [Num of Features] -> (Chan Num , Freq Num, Temp Num)

    """
    Trial_set = Trial_Selector(Features=Data_Set, Sel_index=Test_index)

    ml_trials, ml_labels, ordered_index = Series_Classification_Prep_Pipeline(Trial_set, Offset=Offset,
                                                                              Tr_Length=Tr_Length, labels=labels,
                                                                              label_instructions=label_instructions,
                                                                              onsets=onsets,
                                                                              Feature_Type=Feature_Type, re_break=True)
    return ml_trials, ml_labels, ordered_index


def find_accuracy(prediction: np.ndarray, truth: np.ndarray):
    """"Finds the accuracy of the prediction against the truth values

    Parameters:
    -----------
    prediction: np.ndarry
        Predictions as a ndarrary
    truth: np.ndarray
        The True Values as a ndarray

    Returns:
    --------
    accuracy: float
        The percent predicted correctly
    """
    assert len(prediction) == len(truth), "The two inputs must be equal"
    #     return sum([1 for x, y in zip(prediction, truth) if x==y])/len(truth)
    return sum(prediction == truth) / len(truth)


def find_days_accuracy(predictions, truths):
    """Determines the accuracy across all epochs for that day

    Parameters:
    -----------
    predictions: list
        list of each days predictions [epoch] -> (predictions x 1)
    truths: list
        list of each days true labels [epoch] -> (predictions x 1)
    Returns:
    --------
    mean_accuracy: float
        The mean accuracy of the classifier
    std_err: float
        The mean standard error of the accuracy of the classifier
    """

    epoch_accuracy = []
    for ep_pred, ep_truth in zip(predictions, truths):
        epoch_accuracy.append(find_accuracy(ep_pred, ep_truth))

    mean_acc = np.mean(epoch_accuracy)
    std_err = np.std(epoch_accuracy) / np.sqrt(len(epoch_accuracy))

    return mean_acc, std_err


def find_days_confusion(predictions, truths, labels=None):
    """Calculate the Confusion Matrix for the Day

    :param predictions:
    :param truths:

    labels: list
        list of the classes as interpreted by Scikitlearn
        optional List of labels to index the matrix. This may be used to reorder or select a subset of labels.
        If none is given, those that appear at least once in y_true or y_pred are used in sorted order.

    Returns:
    --------

    """
    confusion_step = []
    for ep_pred, ep_truth in zip(predictions, truths):
        # print('Prediction Values: ', ep_pred, 'True Labels: ', ep_truth)
        if labels:
            confusion_step.append(confusion_matrix(ep_truth, ep_pred, labels=labels).astype(float))
        else:
            confusion_step.append(confusion_matrix(ep_truth, ep_pred).astype(float))

    confusion = np.zeros((np.shape(confusion_step[0])))
    for epoch in confusion_step:
        confusion = confusion + epoch
    return confusion


# TODO: Make this a universal funciton that can be used nomater the context

def predict_by_epoch(Classifier, Features):
    """Test a trained classifier from one day during another day

    Parameters:
    -----------


    Returns:
    --------

    """

    epochs_predictions = []

    for epoch in Features:
        epochs_predictions.append(Classifier.predict(epoch))

    return epochs_predictions


def classify_another_day(Classifier, features, truths, sckit_labels=None):
    """Test a trained classifier from one day on another day and break into its epochs then characterize its behavior

    Parameters:
    -----------
    classifier:
        The Trained classifier to be tested
    features:
    truths:
    scikit_labels: list
        list of labels

    Returns:
    --------
    epoch_predictions:  list
        list of each epochs predictions. [epoch] -> (Predicted Label x 1)
    epoch_truths: list
        list of each epochs true label. [epoch] -> (true Label x 1)
    mean_acc: float
        the mean accuracy of the classifier across all epochs for that day
    std_err: float
        the mean std error of the classifier across all epochs for that day
    confusion:
        Confusion matrix, shape = [n_classes, n_classes]
    """

    epoch_predictions = predict_by_epoch(Classifier, features)  # Predict labels for the entire day

    mean_acc, std_err = find_days_accuracy(epoch_predictions, truths)  # Calculate the mean and standard Error

    confusion = find_days_confusion(epoch_predictions, truths, sckit_labels)  # Calculate the confusion matrix

    return epoch_predictions, truths, mean_acc, std_err, confusion


########################################################################################################################
####################### Code for Visualizing the Onsets for Trained Models ######################################
########################################################################################################################


# def Onset_Detection_Metrics(True_Onsets, Onset_Predictions, Offset: int):
#     """Finds the distance between ONE class's predictions to the nearest true value for ONE Specific Label for ONE Trial
#
#     Parameters:
#     -----------
#     True_Onsets:
#
#     Onset_Predictions:
#
#     Offset: int
#
#
#     Returns:
#     --------
#     Onset_Holder:
#
#     """
#
#     onset_holder = np.zeros([len(Onset_Predictions), 1])
#
#     for onset_ind, onset in enumerate(Onset_Predictions):
#         candidates = np.zeros([len(True_Onsets), 1])
#         for true_index, true_start in enumerate(True_Onsets):
#             candidates[true_index] = onset + Offset - true_start
#         closest_onset = [y for y in candidates if abs(y) == min(abs(candidates))]
#         onset_holder[onset_ind] = closest_onset[0]
#     return onset_holder

def Starts_Extract_Pipeline(All_Labels, Time_Stamps, Label_Instructions):
    """ Pipeline for Extracing all of the Starts of User Selected Labels and
    return them the designated manner.

    Label_Instructions = tells the Function what labels to extract and whether to group them together"""

    Starts = []
    Ends = []

    for i in range(len(Label_Instructions)):
        if type(Label_Instructions[i]) == int or type(Label_Instructions[i]) == str:
            Label_Starts = Label_Focus(Label_Instructions[i], All_Labels, Time_Stamps[0])
            Label_Ends = Label_Focus(Label_Instructions[i], All_Labels, Time_Stamps[1])
        else:
            Label_Starts = Label_Grouper(Label_Instructions[i], All_Labels, Time_Stamps[0])
            Label_Ends = Label_Grouper(Label_Instructions[i], All_Labels, Time_Stamps[1])

        Starts.append(Label_Starts)
        Ends.append(Label_Ends)
    return Starts, Ends


def Onset_Detection_Metrics(True_Onsets, Onset_Predictions, Offset=int):
    """ Finds the distance between ONE class's predictions to the nearest true value
    for ONE Specific Label for ONE Trial

    """
    Onset_Holder = np.zeros([len(Onset_Predictions), 1])
    #     Onset_Holder = []
    #     print np.shape(Onset_Holder)
    for i in range(len(Onset_Predictions)):
        Onset_candidates = np.zeros([len(True_Onsets), 1])
        for j in range(len(True_Onsets)):
            Onset_candidates[j] = Onset_Predictions[i] + Offset - True_Onsets[j]
        Closest_Onset = [Onset_candidates[x] for x, y in enumerate(Onset_candidates) if
                         abs(y) == min(abs(Onset_candidates))]
        #         print Closest_Onset
        Onset_Holder[i] = Closest_Onset[0]
    return Onset_Holder


# Predicted Onset Finder

def Predicted_Onset_Finder(Onset_Predictions, Sel_Label):
    """ Find all of the predictions for a particular Label [Labels are denamed into Label_Instruction Order]"""
    Pred_Onsets = [x for x, y in enumerate(Onset_Predictions) if y == Sel_Label]
    return Pred_Onsets


def Label_Onset_Culmination(One_Classifier, Series_PrePd, All_Dev_Starts, Test_Index, Sel_Label=int, Offset=int):
    """Go over all Test Sets for one Label

    Input:
        [1] One_Classifier: A Specified Trained Classifier
        [2] Series_PrePd: List of Test Sets Feature Extracted and Aligned in Time
                {Series_PrePd is Prepared by the Function kFold_Series_Test}
        [3] All_Dev_Starts: All Starts For All Labels [For Nesting in a Series Labeled Function]
        [4] Test_Index: Index of the Classifiers Test Set
    """

    Full_Hist = np.zeros([1, 1])

    for i in range(len(Test_Index)):
        # Find the Predicted Starts for a particular Label of One Trial
        Syll_predict = Predicted_Onset_Finder(One_Classifier.predict(Series_PrePd[Test_Index[i]]), Sel_Label=Sel_Label)

        print(np.shape(Syll_predict))
        print(np.shape(All_Dev_Starts[Sel_Label][Test_Index[i]]))
        Hist_Components = Onset_Detection_Metrics(All_Dev_Starts[Sel_Label][Test_Index[i]],
                                                  Syll_predict, Offset=Offset)

        Full_Hist = np.concatenate((Full_Hist, Hist_Components), axis=0)
    Full_Hist = np.delete(Full_Hist, 0, 0)  # Delete the First Row (Initialized Row)
    return Full_Hist


def Classifier_Onset_Metrics(One_Classifier, Series_PrePd, All_True_Onsets, Test_Index, Class_Index, Offset=int):
    """Loops over the Test Set for a Single Trained Classifier. One Label at a time

    Inputs:
        [1] One_Classifier: A Specified Trained Classifier
        [2] Series_PrePd: List of Test Sets Feature Extracted and Aligned in Time
                {Series_PrePd is Prepared by the Function kFold_Series_Test}
        [3] All_True_Onsets: All Starts For All Labels [For Nesting in a Series Labeled Function]
        [4] Test_Index: Index of the Classifiers Test Set
        [5] Class_Index: Index of the Labels passed as Label_Instructions

    Output:
            [Number of Labels]-> ndarray(Offsets,1)
    """
    print
    'Under Development'

    Fold_Onsets = []

    for i in range(len(Class_Index)):
        Full_HIST_TEST = Label_Onset_Culmination(One_Classifier=One_Classifier,
                                                 Series_PrePd=Series_PrePd,
                                                 All_Dev_Starts=All_True_Onsets,
                                                 Test_Index=Test_Index,
                                                 Sel_Label=i,
                                                 Offset=Offset)
        Fold_Onsets.append(Full_HIST_TEST)
    return Fold_Onsets


def All_Folds_Onset_Metrics(Data_Set, K_Classifiers, Label_Instructions, All_True_Onsets, Offset=int, Tr_Length=int,
                            Feature_Type='Power', verbose=False):
    ''' Loop over each Fold's Trained Classifier and Find the each Labels Onset Metrics

    Inputs:
        [1] Data_Set:
        [2] K_Classifier: All Trained Classifier Components
        [3] Label_Instructions: Index of the Labels Instructions
        [4] All_True_Onsets: All Starts For All Labels [For Nesting in a Series Labeled Function]

    Output:
            [Dict of Folds] -> [Number of Labels] -> ndarray(Offsets,1)
    '''
    #     Dev_Starts, Dev_Ends = Starts_Extract_Pipeline(Day2_Labels, Day2_Clippings, [1,2,3,4,'I'])

    # Break-up Tuple
    Classifier_d, Test_Index_d = K_Classifiers
    Fold_Hist = dict()

    for i in range(len(Test_Index_d)):  # For Each Fold

        if verbose == True:
            print
            'Fold #: ' + str(i + 1)
        ## Prepare the Series Test_Set
        kFold_Series = Full_Trial_LFP_Clipper(Data_Set, Sel_Motifs=Test_Index_d[i], Num_Freq=13,
                                              Num_Chan=16, Sn_Len=500, Gap_Len=4000)

        # Neural, Sel_Motifs, Num_Freq, Num_Chan, Sn_Len, Gap_Len

        Fold_Hist[i] = Classifier_Onset_Metrics(Classifier_d[i], kFold_Series, All_True_Onsets,
                                                Test_Index_d[i], Class_Index=Label_Instructions,
                                                Offset=Offset)

    return Fold_Hist


import matplotlib.mlab as mlab


def Onset_Histogram(Label_Onsets, Bin_Width=1, Normalize=True, ax=False):
    if ax == False:
        plt.hist(Label_Onsets, bins=len(range(min(Label_Onsets), max(Label_Onsets))) / Bin_Width, normed=Normalize,
                 stacked=False)
        plt.axvline(x=0, linestyle='--', color='black')
    if ax == True:
        ax.hist(Label_Onsets, bins=len(range(min(Label_Onsets), max(Label_Onsets))) / Bin_Width, normed=Normalize,
                stacked=False)
        ax.axvline(x=0, linestyle='--', color='black')


def Fold_Onsets_Histograms(Fold_Onsets, Bin_Width=1, Normalize=True, Repeat=False, ax=False):
    #     fig= plt.figure(figsize=(20,30))
    if Repeat == False:
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle('Onsets for Fold #: %d' % (1), y=1.01, size=30)

    Map = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

    for i in range(len(Fold_Onsets)):
        print
        min(Fold_Onsets[i])
        print
        max(Fold_Onsets[i])
        print
        len(range(min(Fold_Onsets[i]), max(Fold_Onsets[i])))
        ax[Map[i][0], Map[i][1]].hist(Fold_Onsets[i],
                                      bins=((len(range(min(Fold_Onsets[i]), max(Fold_Onsets[i]))))) / Bin_Width,
                                      align='mid', normed=Normalize, stacked=False)
        ax[Map[i][0], Map[i][1]].axvline(x=0, linestyle='--', color='black')
        ax[Map[i][0], Map[i][1]].set_title('Label: %d' % (i))


def Better_Onsets_Histograms(Fold_Onsets, Labels, Bin_Width=1, Normalize=True, Repeat=False, ax=False):
    sizeCorrect = 15
    if Repeat == False:
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle('Onsets for Fold #: %d' % (1), y=1.01, size=30 - sizeCorrect)

    Map = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

    for i in range(len(Fold_Onsets)):
        n, bins, patches = ax[Map[i][0], Map[i][1]].hist(Fold_Onsets[i],
                                                         bins=((len(range(min(Fold_Onsets[i]),
                                                                          max(Fold_Onsets[i]))))) / Bin_Width,
                                                         normed=1, alpha=0.5)

        #         ax[Map[i][0],Map[i][1]].hist(Fold_Onsets[i], bins = ((len(range(min(Fold_Onsets[i]),max(Fold_Onsets[i])))))/Bin_Width, align= 'mid' , normed = Normalize, stacked = False)
        ax[Map[i][0], Map[i][1]].axvline(x=0, linestyle='--', color='black', linewidth=2)

        ax[Map[i][0], Map[i][1]].set_title(str(Labels[i]), size=30 - sizeCorrect)

        ax[Map[i][0], Map[i][1]].set_xlabel('Time [10 ms Binwidth]', fontsize=30 - sizeCorrect)
        #         ax[Map[i][0],Map[i][1]].set_ylabel('Normalized Count', fontsize=30-sizeCorrect)
        ax[Map[i][0], Map[i][1]].tick_params(axis='both', which='major', labelsize=25 - sizeCorrect)
        ax[Map[i][0], Map[i][1]].tick_params(axis='both', which='minor', labelsize=25 - sizeCorrect)


def Better_Onsets_Histograms2(Fold_Onsets, Labels, Bin_Width=1, Normalize=True, Repeat=False, ax=False):
    if Repeat == False:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Onsets for Fold #: %d' % (1), y=1.01, size=30)

    Map = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

    for i in range(len(Fold_Onsets)):
        n, bins, patches = ax[i].hist(Fold_Onsets[i],
                                      bins=((len(range(min(Fold_Onsets[i]), max(Fold_Onsets[i]))))) / Bin_Width,
                                      normed=1, alpha=0.5)

        #         ax[Map[i][0],Map[i][1]].hist(Fold_Onsets[i], bins = ((len(range(min(Fold_Onsets[i]),max(Fold_Onsets[i])))))/Bin_Width, align= 'mid' , normed = Normalize, stacked = False)
        ax[i].axvline(x=0, linestyle='--', color='black', linewidth=4)
        ax[i].set_title(str(Labels[i]), size=30)

        ax[i].set_xlabel('Time [10 ms Binwidth]', fontsize='30')
        ax[i].set_ylabel('Normalized Count', fontsize='30')
        ax[i].tick_params(axis='both', which='major', labelsize=25)
        ax[i].tick_params(axis='both', which='minor', labelsize=25)


def Better_Onsets_Histograms3(Fold_Onsets, Labels, Bin_Width=1, Normalize=True, Repeat=False):
    if Repeat == False:
        fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        fig.suptitle('Onsets for Fold #: %d' % (1), y=1.01, size=30)

    for i in range(len(Fold_Onsets)):
        n, bins, patches = ax[i].hist(Fold_Onsets[i],
                                      bins=((len(range(min(Fold_Onsets[i]), max(Fold_Onsets[i]))))) / Bin_Width,
                                      normed=1, alpha=0.5)

        #         ax[Map[i][0],Map[i][1]].hist(Fold_Onsets[i], bins = ((len(range(min(Fold_Onsets[i]),max(Fold_Onsets[i])))))/Bin_Width, align= 'mid' , normed = Normalize, stacked = False)
        ax[i].axvline(x=0, linestyle='--', color='black', linewidth=4)
        ax[i].set_title(str(Labels[i]))


#     n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
#     # add a 'best fit' line
#     y = mlab.normpdf(bins, mu, sigma)
#     plt.plot(bins, y, 'r--')


# ----------------------------
# Scratch post


def Feature_Dropping_Selector(features, labels, removal_index):
    """Takes Epochs and Labels that are ndarray compatible with scikitlearn and takes away the set designated by removal_index

    Note: This function works by removing the entries for the indexes given by removal_index.
    For example to get the training set you would input the index for the test set for the removal_index variable.

    Parameters:
    -----------
    features: ndarray
        Array that is structured to work with the SciKit-learn Package
        (n_samples, n_features)
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    labels: ndarray
        1-d array of Labels of the Corresponding n_samples
        ( n_samples   x   1 )
    removal_index: list
        list of indexes to be removed from the features input to get the opposite set.


    Returns:
    --------
    sel_set:

    sel_labels:

    sel_set: ndarray
        Array of the selected K-Fold's Opposite corresponding set of the input
        [ch] -> [Freq] -> (Time x Num_Epoxhs)
    sel_labels: ndarray
        1-d array of Labels of the Opposite Corresponding n_samples
        ( n_samples   x   1 )
    """

    sel_set = np.delete(features, removal_index, axis=0)
    sel_labels = np.delete(labels, removal_index, axis=0)
    return sel_set, sel_labels


def Drop_Features(Features, Keys, Desig_Drop):
    """Function for Selectively Removing Columns for Feature Dropping
    Des_igDrop is short for Designated to be Dropped
    """

    Full_Drop = []

    for i in range(len(Desig_Drop)):
        Full_Drop.extend(Keys[Desig_Drop[i]])

    Remaining_Features = np.delete(Features, Full_Drop, axis=1)

    return Remaining_Features, Full_Drop


def make_channel_dict(ordered_index):
    """Creates a Dictionary of the the indexes for each Channel's features in the ordered_index

    Parameters:
    -----------
    ordered_index: list
        Index of Features for Feature Dropping
                            [list] -> (Tuple)
        Power:   [Num of Features] -> (Chan Num , Freq Num)
        Pearson: [Num of Features] -> (Chan Num , Freq Num, Temp Num)

    Returns:
    -------
    channel_dict: dict
        dictionary to be used to remove all features for a single channel
        {Channel: [list of Indexes]}

    """

    channel_dict = {}
    nun_channels = ordered_index[-1][0]+1  # Determine the Number of Channels (Assumes the ordered_index is in order)

    # Iterate over the number of channels
    for chan_focus in range(nun_channels):
        value = []

        # Iterate over the total number of features
        for index in range(len(ordered_index)):
            if ordered_index[index][0]==chan_focus:
                value.append(index)
        channel_dict[chan_focus] = value # Store the list of that Channel's Features to its corresponding Key in the Dict

    return channel_dict





def kfold_wrapper(Data_Set, Data_Labels, k_folds, Class_Obj, verbose=False):
    """ Runs the Clip_Classifcation for Crossfold Validation. (Used for the feature Dropping Code)

    Parameters:
    -----------
    Data_Set: ndarray
        Array that is structured to work with the SciKit-learn Package
        (n_samples, n_features)
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    Data_Labels: ndarray
        1-d array of Labels of the Corresponding n_samples
        ( n_samples   x   1 )
    k_folds: int
        Number of Folds for Cross-Validation
    Class_Obj: class
        classifier object from the scikit-learn package
    verbose: bool
        If True the function will print out useful print statements to inform the user of program's progress

    Returns:
    --------
    mean_acc: int
        the mean accuracy across the folds
    std_err: int
        the standard error across the folds
    classifier_components: tuple
        Tuples containing two Dictionaries with the fold number being the keys (using 0 indexing).
        Their Values are:
            1.) trained Classifier instances and the index of features for their models
                The values are that fold's trained classifier instance of the the CLass_Object Parameter
                (from scikit-learn)
            2.) list of the Test set for the corresponding trained Classifier
       shape =  ({ Fold_Num: Trained_Classifiers }, {Fold_Num: Test_Index})
    confusion: list
        list of each fold's Confusion matrix, shape = [n_classes, n_classes]
    """

    skf = StratifiedKFold(n_splits=k_folds)

    acc = np.zeros(k_folds)
    confusion = []  # Just Added
    # ROC = []  # Just Added too 8/10
    foldNum = 0

    Trained_Classifiers = dict()
    Trained_Index = dict()

    Num_Clippings = np.ones(len(Data_Labels))

    for train, test in skf.split(Num_Clippings, Num_Clippings):
        if verbose:
            print("Fold %s..." % foldNum)
            # print "%s %s" % (train, test)

        print(train)
        train_trials, train_labels = Feature_Dropping_Selector(features=Data_Set, labels=Data_Labels,
                                                               removal_index=test)

        print(test)
        test_trials, test_labels = Feature_Dropping_Selector(features=Data_Set, labels=Data_Labels, removal_index=train)

        acc[foldNum], Trained_Classifiers[foldNum], conf = Clip_Classification(Class_Obj, train_trials,
                                                                               train_labels,
                                                                               test_trials, test_labels,
                                                                               verbose=False)
        Trained_Index[foldNum] = test
        foldNum += 1

        if verbose:
            print(conf)
        confusion.append(conf)

    meanacc = np.mean(acc)
    stderr = np.std(acc) / np.sqrt(k_folds)
    classifier_components = (Trained_Classifiers, Trained_Index)

    if verbose:
        print("cross-validated acc: %.2f +/- %.2f" % (np.mean(acc), np.std(acc)))
    return meanacc, stderr, classifier_components, confusion,


def run_feature_dropping(Data_Set, Data_Labels, ordered_index, Class_Obj, k_folds=2, verbose=False):
    """ Repeatedly trains/test models to create a feature dropping curve (Originally for Pearson Correlation)

    Parameters:
    -----------

    Returns:
    --------

    """

    # 1.) Initiate Lists for Curve Components
    droppingCurve = []
    StdERR = []
    dropFeats = []

    feat_ids = make_channel_dict(ordered_index=ordered_index)

    # 2.) Initiate Variables
    B = len(feat_ids[0])  # Determine the number of columns per dropped feature
    C = len(feat_ids)  # Find the number of Features
    print(B)
    print(C)
    print(np.shape(feat_ids))


    Temp = feat_ids.copy()  # Create a temporary internal *shallow? copy of the index dictionary

    # 3.) Begin Feature Dropping steps
    # Find the first k-Fold Acc.
    first_mean_acc, first_err_bars, _, _ = kfold_wrapper(Data_Set=Data_Set, Data_Labels=Data_Labels, k_folds=k_folds, Class_Obj=Class_Obj, verbose=verbose)


    if verbose == True:
        print("First acc: %s..." % first_mean_acc)
        print("First Standard Error is: %s" % first_err_bars)  ###### I added this for the error bars
    droppingCurve.append(first_mean_acc)  # Append BDF's Accuracy to Curve List
    StdERR.append(first_err_bars)  # Append BDF's StdErr to Curve List

    N = 16 # Number of Channels
    #     N = 200
    while N > 2:  # Decrease once done with development
        IDs = Temp.keys()  # Make List of the Keys(Features)
        print(IDs)

        N = len(IDs)  # keep track of the number of Features
        print(N)
        meanAcc = np.zeros(N)  # Create Container for all of the Means
        ErrBars = np.zeros(N)  # Create Container for all of the StdErrs

        print(N)

        for n in range(0, N):
            Test_Drop = []
            Test_Drop = copy.deepcopy(dropFeats)  # Copy over the Previously Dropped Features
            Test_Drop.append(n)  # add n to the Previously Dropped Features
            Remaining_Features, _ = Drop_Features(Data_Set, feat_ids, Test_Drop)  # Remove selected Features

            # Find the k-Fold Acc.
            meanAcc[n], ErrBars,  _, _ = kfold_wrapper(Data_Set=Remaining_Features,
                                                       Data_Labels=Data_Labels,
                                                       k_folds=k_folds,
                                                       Class_Obj=Class_Obj,
                                                       verbose=verbose)

        dropFeatID = IDs[meanAcc.argmax()]  # Find the Best Dropped Feature (BDF)
        droppingCurve.append(meanAcc.max())  # Append BDF's Accuracy to Curve List
        StdERR.append(ErrBars)  # Append BDF's StdErr to Curve List

        if verbose == True:
            print("Max. acc: %s..." % (meanAcc.max()))
            print("My attempt at finding the Standard Error is: %s" % (ErrBars) ) ###### I added this for the error bars
            print("Dropping Feature %s..." % (dropFeatID))
        dropFeats.append(dropFeatID)  # Append BDF to List of Dropped Features
        del Temp[dropFeatID]  # Delete key for BDF from Temp Dict
    return droppingCurve, StdERR
