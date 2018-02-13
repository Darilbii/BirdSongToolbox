import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, lfilter

import os
import scipy.io as sio
import random

# Command for Importing Process Birdsong Data (Significantly More Flexible)
def Import_Birds_PrePd_Data(bird_id='z020', sess_name='day-2016-06-02'):
    '''Import Pre-prepared (PrePd) Data and its accomponying meta-data into the workspace for analysis

    Note: This data has be prepared using self created Matlab scripts that required handlabeling.
    Make sure that you are using the correct Preping script.

    Parameters:
    -----------
    bird_id: str
        Bird Indentifier to Locate Specified Bird's data folder
    sess_name: str
        Experiment Day to Locate it's Folder

    Returns:
    --------
    Song_LPF_DS_Data: list [Number of Trials]-> [Trial Length (Samples @ 1KHz) x Ch]
        Lowpass Filtered Neural data during Song Trials (300 Hz. Cutoff)
    Song_Audio_Data: list [Number of Trials]-> [Trial Length (Samples @ 30KHz) x 1]
        Audio of Trials, centered on motif
    Numb_Motifs: int
        Number of Motifs in data set
    Silence_LPF_DS_Data: list [Number of Trials]-> [Trial Length (Samples @ 1KHz) x Ch]
        Lowpass Filtered Neural data during Silent Trials (300 Hz. Cutoff)
    Silence_Audio_Data: list [Number of Trials]-> [Trial Length (Samples @ 30KHz) x 1]
        Audio of Silents Trials
    Numb_Sil_Ex: int
        Number of Examples of Silence
    Labels_Quality: list [Number of Trials x 1 (numpy.unicode_)]
        Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
    Labels_Location: list [Number of Trials x 1 (numpy.unicode_)]
        Describes the Location of the Motif in the BOut, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
    Labels_Syl_Drop: list [Number of Trials x 1 (numpy.unicode_)]
        Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
        *** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility***
    '''
    experiment_folder = '/net/expData/birdSong/'
    Prepd_ss_data_folder = os.path.join(experiment_folder, 'ss_data_Processed')
    #     bird_id = 'z020'
    #     sess_name = 'day-2016-06-02'
    Song_File = os.path.join(Prepd_ss_data_folder, bird_id, sess_name, 'Song_LFP_DS.mat')

    ## Song: Store the Low Pass Filtered & Downsampled Neural Data
    Song_LPF_DS_Data = []
    Mat_File = sio.loadmat(Song_File);
    Mat_File_Filt = Mat_File['Song_LFP_DS'];
    Numb_Motifs = len(Mat_File_Filt);

    for i in xrange(0, Numb_Motifs):
        Song_LPF_DS_Data.append(np.transpose(Mat_File_Filt[i, 0]))

    ## Song: Store the Filtered Audio Data
    Song_File = os.path.join(Prepd_ss_data_folder, bird_id, sess_name, 'Song_Audio.mat')

    Song_Audio_Data = []
    Mat_File = sio.loadmat(Song_File);
    Mat_File_Filt = Mat_File['Song_Audio'];

    Song_Audio_Data = []
    for i in xrange(0, Numb_Motifs):
        Song_Audio_Data.append(np.transpose(Mat_File_Filt[i, 0]))

    ## Silence: Store the Low Pass Filtered & Downsampled Neural Data

    Silence_File = os.path.join(Prepd_ss_data_folder, bird_id, sess_name, 'Silence_LFP_DS.mat')

    Silence_LPF_DS_Data = []
    Mat_File = sio.loadmat(Silence_File);
    Mat_File_Filt = Mat_File['Silence_LFP_DS'];
    Numb_Sil_Ex = len(Mat_File_Filt);

    for i in xrange(0, Numb_Sil_Ex):
        Silence_LPF_DS_Data.append(np.transpose(Mat_File_Filt[i, 0]))

    ## Silence: Store the Filtered Audio Data

    Silence_File = os.path.join(Prepd_ss_data_folder, bird_id, sess_name, 'Silence_Audio.mat')

    Silence_Audio_Data = []
    Mat_File = sio.loadmat(Silence_File);
    Mat_File_Filt = Mat_File['Silence_Audio'];

    Silence_Audio_Data = []
    for i in xrange(0, Numb_Sil_Ex):
        Silence_Audio_Data.append(np.transpose(Mat_File_Filt[i, 0]))

    # Store the Different Types of Labels into Seperate Lists

    Labels_File = os.path.join(Prepd_ss_data_folder, bird_id, sess_name, 'Labels_py.mat')

    Labels_Quality = []
    Labels_Location = []
    Labels_Syl_Drop = []

    Mat_File = sio.loadmat(Labels_File);
    Mat_File_Filt = Mat_File['Motif_Labels'];
    Numb_Motifs = len(Mat_File_Filt);

    # Store the Low Pass Filtered & Downsampled Neural Data
    for i in xrange(0, Numb_Motifs):
        Labels_Quality.append(np.transpose(Mat_File_Filt[i, 0]))
        Labels_Location.append(np.transpose(Mat_File_Filt[i, 1]))
        Labels_Syl_Drop.append(np.transpose(Mat_File_Filt[i, 2]))
    return Song_LPF_DS_Data, Song_Audio_Data, Numb_Motifs, Silence_LPF_DS_Data, Silence_Audio_Data, Numb_Sil_Ex, Labels_Quality, Labels_Location, Labels_Syl_Drop



# All Good Motifs, First Motifs, Bad Motifs (No Dropped Syllables), and Bad Motifs (Last Syllable Dropped)
def Grab_Days_Labels(Quality, Location, Dropped_Syl):
    ''' Creates Indexes of Useful Combinations of Hand Labels for Analysis of Recordings

    Parameters:
    -----------
    Quality: list [Number of Trials x 1 (numpy.unicode_)]
        Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
    Location: list [Number of Trials x 1 (numpy.unicode_)]
        Describes the Location of the Motif in the Bout, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
    Dropped_Syl: list [Number of Trials x 1 (numpy.unicode_)]
        Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
        *** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility***

    Returns:
    --------
    Good_Motifz: list
        Index of All Good Motifs, 'Good' is defined as having little noise and no dropped (or missing) syllables
    First_Motifz: list
        Index of All Good First Motifs, this motif is the first motif in a bout and is classified as 'Good'
    Last_Motifz: list
        Index of All Good Last Motifs, this motif is the last motif in a bout and is classified as 'Good'
    Bad_Motifz: list
        Index of All Bad Motifs with no dropped syllables, These motifs have interferring audio noise
    LS_Drop: list
        Index of All Bad Motifs with the last syllable dropped, These motifs are classified as Bad
    All_First_Motifz: list
        Index of All First Motifs in a Bout Regardless of Quality label, This is Useful for Clip-wise (Series) Analysis
    '''

    # 1. All Good Motifs
    # 1.1 Initialize Variables and Memory
    Q_Length = len(Quality)  # Determine the Number of Motifs
    Quality_Holder = np.zeros(Q_Length)  # Allocate Memory for Indexing

    # 1.2 Fill Logical Index for finding Good
    for i in xrange(Q_Length):
        if Quality[i][0] == 'Good':  # Locate Each Good Label
            Quality_Holder[i] = 1  # Create Index of Selected Label

    Good_Motifz = np.where(Quality_Holder == 1)  # Make Index for Where it is True
    Good_Motifz = Good_Motifz[0]  # Weird Needed Step

    # 2. Good First Motifs
    # 2.1 Initialize Variables and Memory
    Location_Length = len(Location)  # Determine the Number of Motifs
    assert Location_Length == Q_Length
    First_Holder = np.zeros(Location_Length)  # Allocate Memory for Indexing

    # 2.2 Fill Logical for Good First Motifs
    for i in xrange(Q_Length):
        if Quality[i][0] == 'Good':
            if Location[i][0] == 'Beginning':  # Locate Desired Label Combination
                First_Holder[i] = 1  # Mark them
    First_Motifz = np.where(First_Holder == 1)  # Create Index of Selected Label
    First_Motifz = First_Motifz[0]  # Weird Needed Step

    # 3. Good Last Motifs
    # 3.1 Initialize Variables and Memory
    Location_Length = len(Location)  # Determine the Number of Motifs
    assert Location_Length == Q_Length
    Last_Holder = np.zeros(Location_Length)  # Allocate Memory for Indexing

    # 3.2 Fill Logical for Good First Motifs
    for i in xrange(Q_Length):
        if Quality[i][0] == 'Good':
            if Location[i][0] == 'Ending':  # Locate Desired Label Combination
                Last_Holder[i] = 1  # Mark them
    Last_Motifz = np.where(Last_Holder == 1)  # Create Index of Selected Label
    Last_Motifz = Last_Motifz[0]  # Weird Needed Step

    # 4. Bad Motifs w/ NO Dropped Syllables
    # 4.1 Initialize Variables and Memory
    D_Syl_Length = len(Dropped_Syl)  # Determine the Number of Motifs
    assert Location_Length == D_Syl_Length
    Bad_NDS_Holder = np.zeros(D_Syl_Length)  # Allocate Memory for Indexing

    # 4.2 Fill Logical for Bad Motifs (No Dropped Syllables)
    for i in xrange(Q_Length):
        if Quality[i][0] == 'Bad':
            if Dropped_Syl[i][0] == 'None':  # Locate Desired Label Combination
                Bad_NDS_Holder[i] = 1  # Mark them
    Bad_Motifz = np.where(Bad_NDS_Holder == 1)  # Create Index of Selected Label
    Bad_Motifz = Bad_Motifz[0]  # Weird Needed Step

    # 5. Bad Motifs w/ LAST Syllable Dropped
    # 5.1 Initialize Variables and Memory
    LS_Drop_Holder = np.zeros(D_Syl_Length)  # Allocate Memory for Indexing

    # 5.2 Fill Logical for Bad Motifs (Last Syllable Dropped)
    for i in xrange(Q_Length):
        if Quality[i][0] == 'Bad':
            if Dropped_Syl[i][0] == 'Last Syllable':  # Locate Desired Label Combination
                LS_Drop_Holder[i] = 1  # Mark them
    LS_Drop = np.where(LS_Drop_Holder == 1)  # Create Index of Selected Label
    LS_Drop = LS_Drop[0]  # Weird Needed Step

    # 6. All First Motifs
    # 6.1 Initialize Variables and Memory
    Location_Length = len(Location)  # Determine the Number of Motifs
    assert Location_Length == Q_Length
    All_First_Holder = np.zeros(Location_Length)  # Allocate Memory for Indexing

    # 6.2 Fill Logical for All First Motifs
    for i in xrange(Q_Length):
        if Location[i][0] == 'Beginning':  # Locate Desired Label Combination
            All_First_Holder[i] = 1  # Mark them
    All_First_Motifz = np.where(All_First_Holder == 1)  # Create Index of Selected Label
    All_First_Motifz = All_First_Motifz[0]  # Weird Needed Step

    return Good_Motifz, First_Motifz, Last_Motifz, Bad_Motifz, LS_Drop, All_First_Motifz

