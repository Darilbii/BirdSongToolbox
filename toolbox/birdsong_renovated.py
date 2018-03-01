# Created by: Daril Brown II
# This Module is a updated version of my first feature extraction exploration steps. It is meant to allow me to revisit this 
# analysis in the future as well as check old findings with ease.

import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider
import ipywidgets as widgets
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import math
from BirdSongToolbox import *

def view_corr_sep(CFT, CFS, Template, CH_Sel, FREQ_SEL, Top, Bottom):
    ''' Plots the Pearson Correlation Coefficient Histogram of Two Classes (Assummed to be Song vs. Silence) with Normalized Mean in solid Line
    
    Parameters:
    -----------
    CFT: list
        (Channel Frequency Trials) Re-organized Neural Data that were used for constructing the Templates
        [Ch]->[Freq]_>[Time(Samples) x Trials]
    CFS: list
        (Channel Frequency Seperated) Re-organized Neural Data to be compared against Template's source
        [Ch]->[Freq]_>[Time(Samples) x Trials]
    Template: list
        Templates (Mean) of Every Frequency Band for Each Channel
        [Ch]->[Freq]_>[Time(Samples) x 1]
    CH_Sel: int
        Selected Recording Channel
    FREQ_SEL: int
        Selected Frequency Band
    Top: list
        List of High Frequency Cuttoffs of Bandpass's used
    Bottom: list
        List of Low Frequency Cutoffs
    
    '''
    # TODO: Add Asserts
    # TODO: Validate it works as intended
    # TODO: Push to Module?
    assert isinstance(CFT, list)
    assert isinstance(CFS, list)
    assert isinstance(Template, list)
    
    Num_Trials = len(CFT[CH_Sel][FREQ_SEL][1,:])

    Song_CorrCoef = np.zeros((1,Num_Trials))
    Silence_CorrCoef = np.zeros((1,Num_Trials))

    for k in range(Num_Trials):
        Song_CorrCoef[0,k], _ = scipy.stats.pearsonr((CFT[CH_Sel][FREQ_SEL][:,k]),(Template[CH_Sel][FREQ_SEL][:,0]))
        Silence_CorrCoef[0,k], _ = scipy.stats.pearsonr((CFS[CH_Sel][FREQ_SEL][:,k]),(Template[CH_Sel][FREQ_SEL][:,0]))

    Feat_Song_men2 = np.mean(Song_CorrCoef)
    Feat_Silence_men2 = np.mean(Silence_CorrCoef)

    plt.figure(figsize = (8,7))
    plt.title('Correlation (Channel %d Frequency Band= %d-%d)' %(CH_Sel, Bottom[FREQ_SEL], Top[FREQ_SEL] ))
    plt.axvline(x=Feat_Song_men2, color = 'coral', linewidth='4')
    plt.axvline(x=Feat_Silence_men2, color = 'r', linewidth='4')
    plt.xlabel('Correlation')
    plt.xlim(-1,1)
    plt.hist(np.transpose(Song_CorrCoef), 20, (-1,1), normed =True, label ='Song', color = 'blue', edgecolor= 'black')
    plt.xticks(rotation='vertical')
    plt.hist(np.transpose(Silence_CorrCoef), 20, range =(-1,1), normed =True, label ='Silence', color = 'green', edgecolor= 'black')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    
def corr_sep_GUI(Song, Silence, All_Temp, Top, Bottom, Num_Chan ):
    '''GUI for Viewing Compared Correlations Plots
    
    Parameters:
    -----------
    CFT: list
        (Channel Frequency Trials) Re-organized Neural Data that were used for constructing the Templates
        [Ch]->[Freq]_>[Time(Samples) x Trials]
    CFS: list
        (Channel Frequency Seperated) Re-organized Neural Data to be compared against Template's source
        [Ch]->[Freq]_>[Time(Samples) x Trials]
    All_Temp: list
        Templates (Mean) of Every Frequency Band for Each Channel
        [Ch]->[Freq]_>[Time(Samples) x 1]
    CH_Sel: int
        Selected Recording Channel
    FREQ_SEL: int
        Selected Frequency Band
    Top: ndarray
        List of High Frequency Cuttoffs of Bandpass's used
    Bottom: ndarray
        List of Low Frequency Cutoffs
    Num_Chan: int
        Number of Channels
    '''
    
    Channel_widget = widgets.IntSlider(value=0, min=0, max= Num_Chan-1,step=1,description='Channel')
    Freq_widget = widgets.IntSlider(value=0, min=0, max= len(Top)-1 ,step=1,description='Freq')
    
    interact(view_corr_sep, CFT =fixed(Song) , CFS = fixed(Silence), Template = fixed(All_Temp), 
             CH_Sel =Channel_widget, FREQ_SEL= Freq_widget, Top = fixed(Top), Bottom= fixed(Bottom))

def Corr_Seperation(Channel_Freq_Song, Channel_Freq_Silence, Match_Test, Num_Channels, Num_Freqs,  Top, Bottom, Trial_Index, Plot= False):
    
    
    ''' Create a Heatmap that seeks to visualize the discernability between instances of Song Activity and Silence
    
    Parameters:
    -----------
    Channel_Freq_Song: list
        Re-organized Neural Data that were used for constructing the Templates
        [Ch]->[Freq]_>[Time(Samples) x Trials]
    Channel_Freq_Silence: list
        Re-organized Neural Data to be compared against Template's source
        [Ch]->[Freq]_>[Time(Samples) x Trials]
    Match_Test: list
        Templates (Mean) of Every Frequency Band for Each Channel
        [Ch]->[Freq]_>[Time(Samples) x 1]
    Num_Channels: int
        Number of recording Channels
    Num_Freq: int
        Number of Frequency Bands
    Top: list
        List of High Frequency Cuttoffs of Bandpass's used
    Bottom: list
        List of Low Frequency Cutoffs
    Trial_Index: list ???
        List of Trials to Be used for Desired Label
    Plot: bool (Optional)
        If set to True it will plot a heatmap of the Normalized results, Defaults to False
    
    Returns:
    --------
    Feat_Seperation_Norm_Corr:  list
        Normalized diffence in mean Pearson Correlation Coefficients of two Classes (The First class is used for the Template)
        [Channels x Freq. Bands]
    Feat_Seperation_Edges_Corr: list
        Distance between Edges of the Histogram of Correlation Coefficients
        [Channels x Freq. Bands]
        
    '''
    #1. Dynamic Handeling of Channels, Frequency Bands, and Trials
    Num_Channels = len(Channel_Freq_Song)
    Num_Trials = len(Channel_Freq_Song[0][0][1,:]) ##### This needs to be made more Dynamics
    Num_Features = len(Channel_Freq_Song[0])
    
    #2. Initiate Memory for each Channel
    Feat_Seperation_Edges_Corr = np.zeros((16, Num_Features)) # For Edges
    Feat_Seperation_Norm_Corr = np.zeros((16, Num_Features)) # For Norm??

    #3. Initiat Seperate List for Song and Silence
    Chan_Corr_Song = []
    Chan_Corr_Silence = []
    

    #4. Meat of Function
    for CH_Sel in xrange(Num_Channels):  
        for FREQ_SEL in xrange(len(Top)):
            #4.1 Create Memory Space
            Song_CorrCoef = np.zeros((1,len(Trial_Index)))
            Silence_CorrCoef = np.zeros((1,len(Trial_Index)))
            Freq_Holder_Song= []
            Freq_Holder_Silence= []

            #4.2 For Each Trial Find the Correlation Coefficients
            for k in xrange(len(Trial_Index)):
                Song_CorrCoef[0,k], _ = scipy.stats.pearsonr((Channel_Freq_Song[CH_Sel][FREQ_SEL][:,k]),(Match_Test[CH_Sel][FREQ_SEL][:,0]))
                Silence_CorrCoef[0,k], _ = scipy.stats.pearsonr((Channel_Freq_Silence[CH_Sel][FREQ_SEL][:,k]),(Match_Test[CH_Sel][FREQ_SEL][:,0]))
        
            #4.3 Find Difference between Edges to Determine Overlap
            Feat_Seperation_Edges_Corr[CH_Sel,FREQ_SEL] = find_edges(np.median(Song_CorrCoef), np.median(Silence_CorrCoef), Song_CorrCoef, Silence_CorrCoef) # Store Edge Overlap Result
        
            #4.4 Normalized Mean Distance:
            # Divide the difference between the mean of Song Corr-Coef and Silence Corr-Coef by the Sum of their Standard Deviations
            # ((Mean of Song Corr-Coef) - (Mean of Silence Corr-Coef))/ (Sum of Standard Deviation of Song Corr-Coef & Silence Corr-Coef)
            Feat_Seperation_Norm_Corr[CH_Sel,FREQ_SEL] = ((np.mean(Song_CorrCoef) - np.std(Silence_CorrCoef))/((np.std(Song_CorrCoef))+(np.std(Silence_CorrCoef))))
            
            
            #4.7 Store Values of Coefficents to List (Each entry is a Different Frequency Bin)
            Freq_Holder_Song.append(Song_CorrCoef)
            Freq_Holder_Silence.append(Silence_CorrCoef)
        #4.8 Store Lists of Frequency Bins to a List (Each entry is a Different Channel)
        Chan_Corr_Song.append(Freq_Holder_Song)
        Chan_Corr_Silence.append(Freq_Holder_Silence)
        
        #5. Optionally Print Results
    if Plot ==True:
        plot_corr_seperation(Feat_Seperation_Norm_Corr, Top, Bottom, Num_Channels, Num_Freqs)
    return Feat_Seperation_Norm_Corr, Feat_Seperation_Edges_Corr,
                                              
                                              
def find_edges(Feat_Song_med_holder, Feat_Silence_med_holder, Song_CorrCoef, Silence_CorrCoef):   
    '''Find Difference between Edges to Determine Overlap
    
    Parameters:
    -----------
    Feat_Song_med_holder: float64
        Median of Song Class
        (1x 14)
    Feat_Silence_med_holder: float64
        Median of Silence Class
        (1x14)
    Song_CorrCoef: ndarray 
        Pearson Coefficients for Song Class
        (1 x Trials)
    Silence_CorrCoef: ndarray
        Pearson Coefficients for Silence Class
        (1 x Trials)
    Return:
    -------
    Result_Holder: float
        A niave approximation of the distance between the boundaries of the Histograms of Pearson Correlation Coefficients
    '''
    if Feat_Song_med_holder >= Feat_Silence_med_holder:
        Result_Holder = np.amin(Song_CorrCoef)- np.amax(Silence_CorrCoef)
    elif Feat_Song_med_holder < Feat_Silence_med_holder:
        Result_Holder = np.amin(Silence_CorrCoef)- np.amax(Song_CorrCoef)

    return Result_Holder
                                              
                                          
def plot_corr_seperation(NormSep_Corr, Top, Bottom, Num_Channels, Num_Features):
    ''' Plot seperation in Pearons Correlation Coefficients in a Heatmap
    Parameter:
    ----------
    NormSep_Corr:  list
        Normalized diffence in mean Pearson Correlation Coefficients of two Classes (The First class is used for the Template)
        [Channels x Freq. Bands]
     Top: list
        List of High Frequency Cuttoffs of Bandpass's used
    Bottom: list
        List of Low Frequency Cutoffs
    Num_Channels: int
        Number of recording Channels
    Num_Features: int
        Number of Frequency Bands
    '''

    X_labels = []

    for i in xrange(len(Top)):
        X_labels.append( str(Bottom[i]) + '-' +  str(Top[i]))

    plt.figure(figsize=(15,15))

    plt.imshow(NormSep_Corr, cmap='hot',aspect='auto', interpolation='nearest', origin='lower')#, vmax=3)##### Account for Scaling
    plt.xlabel('Frequency'), plt.ylabel('Channel')
    plt.title('Normalized Distance between Means')
    plt.yticks(range(Num_Channels), range(Num_Channels)) #Dynamic Control of Number of Freq Bands
    plt.xticks(range(Num_Features), X_labels) #Dynamic Control of Number of Channels
    plt.colorbar()
    

# Channel v. Channel Correlation (Per Freq. Band)
def Chan_v_Chan_Corr(Song_Templates):
    ''' Channel v. Channel Correlation Comparision (per Freq. Band)
    
    Parameters:
    -----------
    Song_Templates:
        Templates (Mean) of Every Frequency Band for Each Channel
        [Ch]->[Freq]_>(Time(Samples) x 1)
    
    Returns:
    --------
    CvC_Corrs: list
        Templates (Mean) of Every Frequency Band for Each Channel
        [Freq]->[Channel v. Channel]  
    '''
    
    # Find Number of Channels and Frequency Bands
    Num_Freqs = len(Song_Templates[0]) # Number of Frequency Bands
    Num_Chans = len(Song_Templates)    # Number of Channels
    
    # Initiate List for all Matrixes
    CvC_Corrs = []
    
    # Iterate over each Combination of Channels for each Correlation Matrixes
    for i in xrange(Num_Freqs):
        CvC_Holder = np.zeros([Num_Chans, Num_Chans])
        for j in xrange (Num_Chans):
            for k in xrange(Num_Chans):
                CvC_Holder[j,k], _ =scipy.stats.pearsonr(Song_Templates[j][i], Song_Templates[k][i])
        CvC_Corrs.append(CvC_Holder)
    return CvC_Corrs

def CvC_Corr_Heatmap(All_CvC_Corr, Selected_Freq, Top, Bottom, Absolute = False):
    '''Function for Visualizing the Channel vs. Channel Heatmap
  
    Parameters:
    -----------
    All_CvC_Corr: list
        Templates (Mean) of Every Frequency Band for Each Channel
        [Freq]->[Channel v. Channel]  
    Selected_Freq: int
        The Selected Frequency Band to be Visualized
    Top: ndarray
        List of High Frequency Cuttoffs of Bandpass's used
    Bottom: ndarray
        List of Low Frequency Cutoffs   
    Absolute: bool
        (Defaults to False)
    
    '''
  
    X_labels = [x +1 for x in xrange(len(All_CvC_Corr[0]))] #Dynamic Control of Number of Channels
    
    Freq_label = [str(Bottom[i]) + '-' +  str(Top[i]) for i in xrange(len(Top))]
    plt.figure(figsize=(10,8))

    if Absolute == False:
        plt.imshow(All_CvC_Corr[Selected_Freq], cmap='seismic',aspect='auto', interpolation='nearest', origin='lower', vmax=1, vmin = -1)##### Account for Scaling
    if Absolute == True:
        plt.imshow(abs(All_CvC_Corr[Selected_Freq]), cmap='hot',aspect='auto', interpolation='nearest', origin='lower', vmax=1, vmin = 0)##### Account for Scaling
    
    plt.xlabel('Channel', size = 16), plt.ylabel('Channel', size = 16)
    plt.title('Correlation between Channels Frequency Band (%s)' %(Freq_label[Selected_Freq]), size=25)
    plt.yticks(range(len(X_labels)), X_labels, size = 12)
    plt.xticks(range(len(X_labels)), X_labels, size = 12)
    plt.colorbar()

## Potentially Add Function that RE-organize the Plot into order of Shanks and Depth
def CvC_Gui(CvC_Corr_Results, Top, Bottom):
    ''' Interactively View Channel v. Channel Correlation Heatmap
    
    Parameters:
    -----------
    All_CvC_Corr: list
        Templates (Mean) of Every Frequency Band for Each Channel
        [Freq]->[Channel v. Channel]  
    Top: ndarray
        List of High Frequency Cuttoffs of Bandpass's used
    Bottom: ndarray
        List of Low Frequency Cutoffs   
    '''
    
    Num_Bands = len(CvC_Corr_Results)
    
    Freq_widget = widgets.IntSlider(value=0, min=0, max=Num_Bands-1,step=1,description='Freq Band Width')
    
    Absolute_widget= widgets.ToggleButton( value=True,
                                          description='Absolute Value',
                                          disabled=False,
                                          button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                          tooltip='Description',
                                          icon='check')

    
    interact(CvC_Corr_Heatmap, All_CvC_Corr= fixed(CvC_Corr_Results), Selected_Freq= Freq_widget, Top= fixed(Top), Bottom= fixed(Bottom) , Absolute = Absolute_widget) 

    
# Code for Importing the Geometry of the Probe [Copied from Spatial Mapping Development 6/4/17]
## Need to Make a Program that Reads the .prb (Text File) to Get the Locations [Develop this on Spatial Development]

def Import_Probe_Geometry(bird_id = 'z020', sess_name = 'day-2016-06-02'):
    '''Import the .npy files that seem to be the probe's geometry (Not Everyday has these Files)
    
    Parameters:
    -----------
    bird_id: str
        Bird Indentifier to Locate Specified Bird's data folder
    sess_name: str
        Experiment Day to Locate it's Folder
        
    Returns:
    --------
    Channel_Locations: np.memmap
        Coordinants of contact points
        (Number of Contact Points by (|X coordinate | Y coordinate|))
    Channel_map: np.memmap
        Identities of the Contact Points
        (1 x Number of Contact Points)
    '''

    experiment_folder = '/net/expData/birdSong/'
    ss_data_folder = os.path.join(experiment_folder, 'ss_data')
    kwd_file_folder = os.path.join(ss_data_folder, bird_id, sess_name)
    kwd_files = [f for f in os.listdir(kwd_file_folder) if f.endswith('.kwd')]
    assert(len(kwd_files)==1)
    kwd_file = kwd_files[0]
    print kwd_file # Sanity Check to Make Sure You are working with the Correct File
    
    Location_files = [f for f in os.listdir(kwd_file_folder) if f.endswith('.npy')]
    
    Location_files
    
    Chan_Loc = 'channel_positions.npy'
    Map_Loc = 'channel_map.npy'
    
    Channel_Locations = np.load(os.path.join(kwd_file_folder, Chan_Loc), mmap_mode='r')
    Channel_map = np.load(os.path.join( kwd_file_folder, Map_Loc), mmap_mode='r')
    
    return Channel_Locations, Channel_map

# Code for Plotting the Probe Geometry

def Plot_Geometry(Ch_Locations, Ch_map, bird_id):
    '''
    Parameters:
    -----------
    Channel_Locations: np.memmap
        Coordinants of contact points
        (Number of Contact Points by (|X coordinate | Y coordinate|))
    Channel_map: np.memmap
        Identities of the Contact Points
        (1 x Number of Contact Points)
    bird_id: str
        Bird Indentifier to Locate Specified Bird's data folder
    '''
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    A = Ch_Locations[:,0]
    B = Ch_Locations[:,1]

    plt.scatter(A,B)
    
    for i, txt in enumerate(A):
        ax.annotate(Ch_map[0][i], (A[i],B[i]), size =12)
    
    plt.grid()
    plt.title('Bird ID: ' + str(bird_id))
    plt.show()
# g

