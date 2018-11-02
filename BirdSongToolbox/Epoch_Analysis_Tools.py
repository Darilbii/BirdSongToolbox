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

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier


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
        (Note: that the Starts Argument must be converted to the 1 KHz Sampling Frequency)
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


def Find_Power(Features, Pow_Method='Basic'):
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


def Pearson_Coeff_Finder(Features, Templates, Slow=False):
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


def Pearson_Extraction(Clipped_Trials, Templates):
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
    #     Num_Temps = len(Features[0][0][0,:]) # Number of Templates
    NT = len(Features[0][0][:, 0])  # Number of Trials
    # Create Initial Array
    column_index = []
    ordered_trials = np.zeros((NT, 1))  # Initialize Dummy Array
    channel_count = 0
    # Channel Based Ordering
    for channel in Features:  # Over all Channels
        frequency_count = 0
        for frequency in channel:  # For Range of All Frequency Bins
            ordered_trials = np.concatenate((ordered_trials, frequency), axis=1)
            for temps in range(len(frequency[0, :])):
                # TODO: Refactor Pearson_ML_Order to run faster
                # ordered_trials = np.concatenate((ordered_trials, np.reshape(frequency[:, temps], (NT, 1))), axis=1)
                # ordered_trials = np.concatenate(ordered_trials, np.transpose(frequency), axis=1)
                universal_index = (channel_count, frequency_count, temps)  # Tuple contains (Channel #, Freq Band #)
                column_index.append(universal_index)  # Append Index Tuple in Column Order
            frequency_count += 1
        channel_count += 1
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
        extracted_power.append(Find_Power(label))
    return extracted_power


def ml_order_power(Extracted_Features):
    """Restructure Extracted Power Features to a ndarray structure usable to SciKit-Learn: (n_samples, n_features)

    Parameters:
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
    """Reorganizes the Extracted Features into a Useful Machine Learning Format

    Parameters:
    -----------

    Returns:
    --------
    Ordered_Trials:

    Column_Index:
        ?????(Ch, freq_trials)???? Not Sure!
        Output Shape [Number of Examples vs. Number of Features]
    """
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
def Series_LFP_Clipper(Features, Offset=int, Tr_Length=int):
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
                    if starts[trials][ex] - Offset - Tr_Length >= 0:
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


def Classification_Prep_Pipeline(Full_Trials, All_Labels, Time_Stamps, Label_Instructions, Offset=int, Tr_Length=int,
                                 Feature_Type=str, Temps=None, Slide=None, Step=False):
    """

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
    :param Feature_Type:
    :param Temps:
    :param Slide:
    :param Step:
    :return:
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
        if Temps == None:
            Pearson = Pearson_Extraction(Clips, Temps_internal)
        if Temps != None:
            Pearson = Pearson_Extraction(Clips, Temps)

        ML_Trials, ML_Labels, Ordered_Index = Pearson_ML_Order_Pipeline(Pearson)

        if Temps == None:
            return ML_Trials, ML_Labels, Ordered_Index, Temps_internal

    return ML_Trials, ML_Labels, Ordered_Index


def Series_Classification_Prep_Pipeline(Features, Offset: int, Tr_Length: int, Feature_Type: str, labels: list, Temps=None):
    """

    Parameters:
    -----------
    Features: list
        [Ch]->[Freq]->(Time Samples x Trials)
    Offset: int

    Tr_Length: int

    Feature_Type: int

    Temps:

    Returns:
    --------
    full_trial_teatures: list
        [Epoch]->[Samples/Time x Features]

    """
    Series_Trial = Series_LFP_Clipper(Features, Offset=Offset, Tr_Length=Tr_Length)

    if Feature_Type == 'Power':
        series_power = Find_Power(Series_Trial)
        series_ordered, series_labels, ordered_index = series_power_ml_order_pipeline(series_power, labels=labels)

    elif Feature_Type == 'Pearson':
        series_pearson = Pearson_Coeff_Finder(Series_Trial, Temps)
        series_ordered, series_labels, ordered_index = series_pearson_ml_order_pipeline(series_pearson,
                                                                                        labels=labels)

    elif Feature_Type == 'Both':
        series_power = Find_Power(Series_Trial)
        series_pearson = Pearson_Coeff_Finder(Series_Trial, Temps)
        series_ordered, series_labels, ordered_index = series_both_order_pipeline(extracted_features_power=series_power,
                                                                                  extracted_features_pearson=series_pearson,
                                                                                  labels=labels)


    else:
        print(" You didn't input a Valid Feature Type")
        return

    full_trial_features = []
    trial_length = len(Features[0][0][:, 0]) - Offset - Tr_Length

    # Break the long time series back into the Constituent Epochs
    for i in range(len(Features[0][0][0, :])):
        full_trial_features.append(series_ordered[trial_length * (i):trial_length * (i + 1), :])

    return full_trial_features, series_labels, ordered_index


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

    Returns:
    --------


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


def Label_Selector(Labels, Sel_index):
    """This Function allows you to easily parse out specific Trial's Labels for K-Fold validation
    and Test Set Seperation
    """

    sel_labels = []
    for i in range(len(Sel_index)):
        sel_labels.append(Labels[Sel_index[i]])
    return sel_labels


def Convienient_Selector(Features, Labels, Starts, Sel_index):
    """Abstractly reorganizes the list of Epochs and Labels to ndarray compatible with scikitlearn

    :param Features:
    :param Labels:
    :param Starts:
    :param Sel_index:
    :return:
    """
    sel_set = Trial_Selector(Features=Features, Sel_index=Sel_index)
    sel_labels = Label_Selector(Labels, Sel_index=Sel_index)
    sel_starts = Label_Selector(Starts, Sel_index=Sel_index)
    return sel_set, sel_labels, sel_starts





#### NEED TO AD OPTIONAL IF STATETMENT HANDLING FOR RETURNING THE TEMPLATES FOR SERIES CLASSIFICATION

def Clip_KFold(Class_Obj, Data_Set, Data_Labels, Data_Starts, Label_Instructions, Offset=int, Tr_Length=int,
               Feature_Type=str, K=4, Slide=None, Step=False, verbose=False):
    """

    :param Class_Obj:
    :param Data_Set:
    :param Data_Labels:
    :param Data_Starts:
    Label_Instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Offset = int
        The number of samples away from the true onset to Grab for ML Trials (Can be Before or After)
    Tr_Length=int
        Number of Samples to use for Features
    :param Feature_Type:
    :param K:
    :param Slide:
    :param Step:
    :param verbose:

    :return:

    """
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

        # if Feature_Type != 'Pearson':
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
            ml_train_trials, ml_train_labels, train_ordered_index, Temps_int = Classification_Prep_Pipeline(train_set,
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
                                                                                              Temps=Temps_int,
                                                                                              Slide=Slide,
                                                                                              Step=Step)

        acc[foldNum], Trained_Classifiers[foldNum], C = Clip_Classification(Class_Obj, ml_train_trials, ml_train_labels,
                                                                            ml_test_trials, ml_test_labels,
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

###
# Series Analysis Code

def series_pearson_ml_order_pipeline(Extracted_Features, labels):
    """ series equvalent to pearson_ml_order_pipeline

    :param Extracted_Features:
    :return:

    """
    ml_ready = np.zeros((1, (len(Extracted_Features[0]) * len(Extracted_Features[0][0]) * len(Extracted_Features))))
    ml_labels = np.zeros((1, 1))
    for i in range(len(Extracted_Features)):
        ordered_trials, ordered_index = Pearson_ML_Order(Extracted_Features[i])
        ml_ready = np.concatenate((ml_ready, ordered_trials), axis=0)

        dyn_labels = labels[i]
        ml_labels = np.concatenate((ml_labels, dyn_labels), axis=0)

    ml_ready = np.delete(ml_ready, 0, 0)
    ml_labels = np.delete(ml_labels, 0, 0)
    return ml_ready, ml_labels, ordered_index


def series_power_ml_order_pipeline(Extracted_Features, labels):
    """

    :param Extracted_Features:
    :return:
    """

    ml_ready = np.zeros((1, (len(Extracted_Features[0]) * len(Extracted_Features[0][0]))))
    ml_labels = np.zeros((1, 1))
    for i in range(len(Extracted_Features)):
        ordered_trials, ordered_index = ML_Order(Extracted_Features[i])
        ml_ready = np.concatenate((ml_ready, ordered_trials), axis=0)

        dyn_labels = labels[i]
        ml_labels = np.concatenate((ml_labels, dyn_labels), axis=0)

    ml_ready = np.delete(ml_ready, 0, 0)
    ml_labels = np.delete(ml_labels, 0, 0)
    return ml_ready, ml_labels, ordered_index


def series_both_order_pipeline(extracted_features_power, extracted_features_pearson, labels):
    """

    :param extracted_features_power:
    :return:
    """

    ml_ready = np.zeros((1, (len(extracted_features_power[0]) * len(extracted_features_power[0][0]))))
    ml_labels = np.zeros((1, 1))
    for i in range(len(extracted_features_power)):
        # Power Re-Ordered
        ordered_trials_pow, ordered_index_pow = ML_Order(extracted_features_power[i])
        ml_ready = np.concatenate((ml_ready, ordered_trials_pow), axis=0)

        # Pearson Re-Ordered
        ordered_trials_pears, ordered_index_pears = Pearson_ML_Order(extracted_features_pearson[i])
        ml_ready = np.concatenate((ml_ready, ordered_trials_pears), axis=1)

        dyn_labels = labels[i]
        ml_labels = np.concatenate((ml_labels, dyn_labels), axis=0)

    ml_ready = np.delete(ml_ready, 0, 0)
    ml_labels = np.delete(ml_labels, 0, 0)
    ordered_index = ordered_index_pow + ordered_index_pears
    return ml_ready, ml_labels, ordered_index


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
    :return:
    """

    starts = Onsets[0]
    ends = Onsets[1]
    sel_set = Trial_Selector(Features=Features, Sel_index=Sel_index)
    sel_labels = Label_Selector(Labels, Sel_index=Sel_index)
    sel_starts = Label_Selector(starts, Sel_index=Sel_index)
    sel_ends = Label_Selector(ends, Sel_index=Sel_index)
    return sel_set, sel_labels, (sel_starts, sel_ends)




def series_clip_kFold(Class_Obj, Data_Set, Data_Labels, Data_Onsets, Label_Instructions, Offset=int, Tr_Length=int,
                      Feature_Type=str, k_folds=4, verbose=False):
    """

    :param Class_Obj:
    :param Data_Set:
    :param Data_Labels:
    :param Data_Starts:
    :param Label_Instructions:
    :param Offset:
    :param Tr_Length:
    :param Feature_Type:
    :param k_folds:
    :param Slide:
    :param Step:
    :param verbose:

    :return:

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
        train_set, train_labels, train_onsets = Series_Convienient_Selector(Data_Set, Data_Labels, Data_Onsets[0],
                                                                            train)

        print(test)
        test_set, test_labels, test_onsets = Series_Convienient_Selector(Data_Set, Data_Labels, Data_Onsets[0], test)

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
                                                                                   Temps=templates, labels=train_labels)

        ml_test_trials, ml_test_labels, test_ordered_index = Series_Classification_Prep_Pipeline(Features=test_set,
                                                                                 Offset=Offset,
                                                                                 Tr_Length=Tr_Length,
                                                                                 Feature_Type=Feature_Type,
                                                                                 Temps=templates, labels=test_labels)


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
    return mean_acc_nb, std_err_nb, classifier_components, confusion,
