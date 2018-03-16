# Code For Importing .prb files and reading them to make something useful
# TODO: THIS ISN'T READY FOR FULL USE

import os
import pickle
from six import exec_
    
def Prep_Probe_Geometry(bird_id = 'z020', date ='day-2016-06-03'):
    '''Basic Function for Importing the .prb File and Otputing a useful Format For it
    
    Important:
    ----------
    The PRB File is a Python script that describes the probe used for the experiment: its geometry and topology. 
    It must define a channel_groups variable, which is a list where each item is a dictionary 
    with the following keys:
        - channels
        - graph
        - geometry
    Note: the channel order used in the dat file is specified in the channels property (the order matters).
    
    Parameters:
    -----------
    bird_id: str
        Bird Indentifier to Locate Specified Bird's data folder
    date: str
        Experiment Day to Locate it's Folder
        
    Returns:
    --------
    Chan_groups: dict
        dictionary of containing all information from the PRB File(s) [Code handles all combinations of Organization]
    '''
    
    # Basic Setup for path Creation
    experiment_folder = '/net/expData/birdSong/'
    ss_data_folder = os.path.join(experiment_folder, 'ss_data')
    
    prb_file_folder = os.path.join(ss_data_folder, bird_id, date)
    
    prb_files = [f for f in os.listdir(prb_file_folder) if f.endswith('.prb')]
    assert len(prb_files) > 0, '''.prb file doesn't exist in folder '''  
    
    print prb_files[0] # To see if printed in order
    print len(prb_files)
    
    Chan_groups = {}
    for i in xrange(len(prb_files)):
        with open(os.path.join(prb_file_folder, prb_files[i]), 'r') as f:
            contents = f.read()
        metadata = {}
        exec_(contents, {}, metadata)
        metadata = {k.lower(): v for (k, v) in metadata.items()}
        Chan_groups[i]= metadata['channel_groups']
        f.close()

    return Chan_groups

def get_chan_geometry(Probe):
    '''Sifts Through prb File to Create Dictionary of Each Channels Position
    
    Important:
    ----------
    The PRB File is a Python script that describes the probe used for the experiment: its geometry and topology. 
    It must define a channel_groups variable, which is a list where each item is a dictionary 
    with the following keys:
        - channels
        - graph
        - geometry
    Note: the channel order used in the dat file is specified in the channels property (the order matters).
    
    Parameters:
    -----------
    Probe: dict
        Dictionary organization of the probe
        
    Returns:
    --------
    geometry: dict
        Dictionary of the Probe Geometry
        {Channel Number: tuple(X-Coordinate, Y-Coordinate)}
    

    '''
    geometry = {}
    First_Level = Probe.keys()
    for i in xrange(len(First_Level)): # Incase of Multiple Probe Files
        Second_Level = Probe[First_Level[i]].keys()
        for j in xrange(len(Second_Level)):
            geometry.update({k: v for (k,v) in Probe[First_Level[i]][Second_Level[j]]['geometry'].items()})

    return geometry

