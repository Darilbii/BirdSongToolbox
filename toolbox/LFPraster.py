# Module for Quickly Visualizing and Exploring Data Graphically
# Author: Daril Brown

## This Module is Still in Development ##

# Function to View Trials easily based on Index:

# TODO:  Need to Document Code!!
# Based on Feature_GUI Created when I first Joined Lab. 

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider
import ipywidgets as widgets

# Chosen_Index & Index
#*Deprecated overall Chosen
def index_view(Neural, Audio, Chosen_Index, Index, Freq, Channel, Top, Bottom, Tr_Len, Gap_Len, Plot_Type= 'Raster', Marker = False, Mark = 0 ):
    ''' Overall Ploting Function for Exploring the Context Features of LFP. Plots Overlaping trials of LFP based on Combination of Hand Labels
    
    Parameters:
    -----------
        Neural, 
        Audio, 
        Chosen_Index, 
        Index, 
        Channel, 
        Top, 
        Bottom, 
        Tr_len, 
        Gap_Len, 
        Plot_Type
    Returns:
    --------
    '''
    # Input: Neural, Chosen, Index, Top, Bottom
    # Sn_Len, Tr_Len, Designated Index, Top, Bottom, 

    # 3: Create Figure
    fig, ax= plt.subplots(2,1, figsize=(15,10))
    fig.suptitle('LFP of Channel %d during all Motifs'%(Channel),y= 1.12, size= 30)
    colors = ['k','g','r','y','c','m', 'maroon','brown', 'indigo','lime']
    
    Chosen_Index = list(Chosen_Index)
    Index_List = handle_index_list(Chosen_Index, Index)

    if type(Index_List)!=list:
        Problem, Name = Index_List
        print Problem + Name
        return
    
    
    # 4: Plot Audio
    if len(Chosen_Index)==1:
        plot_audio(Audio, Chosen_Index, Index_List[0], Channel, Tr_Len, Gap_Len, ax = ax, colors = colors)
    if len(Chosen_Index)!=1:
        plot_audio_DEV(Audio, Chosen_Index, Index_List, Channel, Tr_Len, Gap_Len, ax = ax, colors = colors)
    
    if Plot_Type=='Raster':
        if len(Chosen_Index)==1:
            plot_raster_single(Neural, Chosen_Index, Index_List, Freq, Channel, Top, Bottom,Tr_Len, Gap_Len, ax = ax, colors = colors)
        if len(Chosen_Index)!=1:
            plot_raster(Neural, Chosen_Index, Index_List, Freq, Channel, Top, Bottom,Tr_Len, Gap_Len, ax = ax, colors = colors)
    
    if Marker== True:
        HandMarker(Mark, ax = ax)
        Time_Markers(Tr_Len, ax = ax)
        

        
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
    

def handle_index_list(Chosen_Index, Index):
    '''Creates List of the Chosen Label Indexs for Viewing'''
    Index_List = []
    
    for i in xrange(len(Chosen_Index)):
        Index_List.append(Index[Chosen_Index[i]])
        if len(Index[Chosen_Index[i]]) < 1:
            return ('Empty Index Named:', Chosen_Index[i])
    return Index_List
        
def plot_audio(Audio, Chosen, Index, Channel,Tr_Len, Gap_Len, ax, colors):
    '''Plot Overalapping Audio'''
    # 4: Plot Audio
    ax[0].set_ylabel('Frequency [Hz]')
    
    for i in range(len(Index) -1): # For Range of Indexed Motifs minus 1
        ax[0].plot(Audio[Index[i]][((Gap_Len/2)-Tr_Len)*30:((Gap_Len/2)+(Tr_Len*2))*30,0], linestyle='-') 
    # Save Last Index to Handle the Labels for Legend
    ax[0].plot(Audio[Index[len(Index)-1]][((Gap_Len/2)-Tr_Len)*30:((Gap_Len/2)+(Tr_Len*2))*30,0], linestyle='-', )
    ax[0].set_title('Pressure Wave of Motif' )
    ax[0].set_ylabel('Arbitruary Units')
    ax[0].set_xlim(0, (Tr_Len*3)*30)
    
def plot_audio_DEV(Audio, Chosen, Index, Channel,Tr_Len, Gap_Len, ax, colors):
    '''Plot Overalapping Audio'''
    # 4: Plot Audio
    ax[0].set_ylabel('Frequency [Hz]')
    
    for l in xrange(0, len(Chosen)):     # Num of Selected Indexes Must be Dynamic
        for i in range(len(Index[l]) -1): # For Range of Indexed Motifs minus 1
            ax[0].plot(Audio[Index[l][i]][((Gap_Len/2)-Tr_Len)*30:((Gap_Len/2)+(Tr_Len*2))*30,0], linestyle='-', color = colors[l]) 
        # Save Last Index to Handle the Labels for Legend
        ax[0].plot(Audio[Index[l][len(Index[l])-1]][((Gap_Len/2)-Tr_Len)*30:((Gap_Len/2)+(Tr_Len*2))*30,0], linestyle='-',  color = colors[l])
    ax[0].set_title('Pressure Wave of Motif' )
    ax[0].set_ylabel('Arbitruary Units')
    ax[0].set_xlim(0, (Tr_Len*3)*30)

    # Freq Constant, Selecting Multiple Indexes
def plot_raster(Neural, Chosen, Index, Freq, Channel, Top, Bottom,Tr_Len, Gap_Len, ax, colors):
    ''' Plots a overlapping view of LFP Activity (Overalapping Indexs)
    ''' 

    
    # 5: Plot Features
    for l in xrange(0, len(Chosen)):     # Num of Selected Indexes Must be Dynamic
        for i in xrange(len(Index[l]) -1): # For Range of Indexed Motifs minus 1
            ax[1].plot(Neural[Index[l][i]][Channel][(Gap_Len/2)-Tr_Len:(Gap_Len/2)+(Tr_Len*2), Freq], color= colors[l], linestyle='-') 
        # Save Last Index to Handle the Labels for Legend
        ax[1].plot(Neural[Index[l][-1]][Channel][(Gap_Len/2)-Tr_Len:(Gap_Len/2)+(Tr_Len*2), Freq], color= colors[l], linestyle='-', label = Chosen[l]) 
    ax[1].set_xlim(0, (Tr_Len*3))

def plot_raster_single(Neural, Chosen, Index, Freq, Channel, Top, Bottom,Tr_Len, Gap_Len, ax, colors):
    ''' Plots a overlapping view of LFP Activity (Overalapping Indexs)
    ''' 

    
    # 5: Plot Features
    for i in xrange(len(Index[0])): # For Range of Indexed Motifs minus 1
        ax[1].plot(Neural[Index[0][i]][Channel][(Gap_Len/2)-Tr_Len:(Gap_Len/2)+(Tr_Len*2), Freq], linestyle='-') 
    ax[1].set_xlim(0, (Tr_Len*3))
    
def plot_raster_Frequency(Neural, Chosen, Index, Channel, Top, Bottom, Tr_Len, Gap_Len, ax, colors):
    ''' Plots a overlapping view of LFP Activity (Overalaping Frequecny Bands)
    
    # Chosen: Chosen Frequency Bands
    ''' 

    
    # 5: Plot Features
    for l in xrange(0, len(Chosen)):     # Num of Selected Features Must be Dynamic
        for i in range(len(Index) -1):   # For Range of Indexed Motifs minus 1
            ax[1].plot(Neural[Index[i]][Channel][(Gap_Len/2)-Tr_Len:(Gap_Len/2)+(Tr_Len*2),int(Chosen[l])], color= colors[l], linestyle='-') 
        # Save Last Index to Handle the Labels for Legend
        ax[1].plot(Neural[Index[len(Index)-1]][Channel][(Gap_Len/2)-Tr_Len:(Gap_Len/2)+(Tr_Len*2), int(Chosen[l])],color= colors[l], linestyle='-', label ='Frequency Band='+ str(Bottom[Chosen[l]]) + '-' + str(Top[Chosen[l]])) 
    ax[1].set_xlim(0, (Tr_Len*3))

def HandMarker(Mark, ax):
    ''' Handles Gui Marker for Visualization'''
    assert isinstance(Mark, int)
    
    ax[0].axvline(x = Mark*30, color = 'blue')
    ax[1].axvline(x = Mark, color = 'blue')

def Time_Markers(Tr_Len, ax):
    '''Make Markers for Easy Viewing of Overlapping Plots'''
    ax[0].axvline(x = Tr_Len*30, color = 'red')
    ax[0].axvline(x = Tr_Len*2*30, color = 'red')
    ax[1].axvline(x = Tr_Len, color = 'red')
    ax[1].axvline(x = Tr_Len*2, color = 'red')
    
def Overlap_GUI(Neural, Audio, Num_Chan, Top, Bottom, Tr_Len, Gap_Len, INSTANCE):
    
    
    
    Chan_widget = widgets.IntSlider(value=7, min=0, max= Num_Chan-1, step=1,
                      description='Channel:',
                      disabled=False,
                      continuous_update=False,
                      orientation='horizontal',
                      readout=True,
                      readout_format='d'
                     )
    Freq_widget = widgets.IntSlider(value=0, min=0, max= len(Top)-1, step=1,
                      description='Frequency Band:',
                      disabled=False,
                      continuous_update=False,
                      orientation='horizontal',
                      readout=True,
                      readout_format='d'
                     )
    
    Index_widget = widgets.Dropdown(options={'One': 1, 'Two': 2, 'Three': 3},
                                    value=2,
                                    description='Number:',
                                )
    
    HandMark_Widget = widgets.IntSlider(value=0, min=0, max= (Tr_Len*3), step=1,
                      description='User Marker:',
                      disabled=False,
                      continuous_update=True,
                      orientation='horizontal',
                      readout=True,
                      readout_format='d'
                     )
    
    Mark_widget = widgets.ToggleButton(
                                        value=True,
                                        description='With Markers',
                                        disabled=False,
                                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Description',
                                        icon='check'
                                    )
    
    Index_widget = widgets.SelectMultiple(options=['Good Motifs',
                                                   'Good First Motifs',
                                                   'Good Middle Motifs',
                                                   'Good Last Motif',
                                                   'All First Motifs',
                                                   'All Last Motifs',
                                                   'Last Syllable Dropped',  
                                                   'Bad Full Motifs'],
                                          value = ['Good Motifs'],
                                          description= 'Hand Labels Used:', 
                                          
                                          button_style='', 
                                          disabled=False)
    
    INDEX = make_index_dict(INSTANCE) # Create Index of Hand Labels
    
# ,Bad_Motifs, .LS_Drop, self.Last_Motifs, .All_First_Motifs, self.First_Motifs, self.Good_Motifs
    
    interact(index_view, 
             Neural= fixed(Neural),
             Audio= fixed(Audio),
             Chosen_Index = Index_widget,
             Chosen= Freq_widget, 
             Index= fixed(INDEX),
             Freq = Freq_widget,
             Channel=Chan_widget, 
             Top= fixed(Top), 
             Bottom= fixed(Bottom), 
             Tr_Len=fixed(Tr_Len), 
             Gap_Len = fixed(Gap_Len),
             Marker = Mark_widget, 
             Plot_Type= fixed('Raster'),
             Mark = HandMark_Widget
            )
    
    
    
def make_index_dict(INSTANCE):
    label_index = {'Good First Motifs': list(INSTANCE.First_Motifs),
                   'Good Motifs': list(INSTANCE.Good_Motifs),
                   'Bad Full Motifs': list(INSTANCE.Bad_Motifs), 
                   'Last Syllable Dropped': list(INSTANCE.LS_Drop), 
                   'Good Last Motif': list(INSTANCE.Last_Motifs),
                   'All First Motifs': list(INSTANCE.All_First_Motifs),
                  'Good Middle Motifs': list(INSTANCE.Good_Mid_Motifs),
                  'All Last Motifs': list(INSTANCE.All_Last_Motifs)}
    return label_index

    # TODO: Multi-Channel View
    # TODO: Multi Choose Indexing, Is Multi add Freq_Bands Still Necesary
    # TODO: What I Need: Easily View Different Channel, Different Frequency Bands, Different Indexs
