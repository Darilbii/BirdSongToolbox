import numpy as np
import os 
import h5py


def ask_data_path(start_path, focus: str):
    """ Asks the User to select data they wish to select for adding to a file path

    Parameters:
    -----------
    start_path:
        path to look for data selection options
    focus: str

    Returns:
    -------
    selection: str
        the selected path addition the user selected to be returned
    """

    escape_words = ['quit', 'exit', 'stop']

    options = os.listdir(start_path)
    print(*options, sep='\n')
    selection = str(input(f'Please Select a {focus}:'))

    # Make sure that the selected value is available
    incorrect_data = True
    while incorrect_data:
        if selection.lower() in escape_words:
            return None
        elif selection not in options:
            print(f'{focus} not available')
            print(*options, sep='\n')
            selection = str(input(f'Now Please Select a {focus}: '))
        else:
            print(f'{focus} Selected')
            incorrect_data = False
    # Folder for the bird

    return selection


def get_data_path(day_folder, file_type: str):
    """ Asks the User to select the data file the user wishes to add to the file path

    Parameter
    ---------
    day_folder:
        path to the day's folder the user selects
    file_type: str
        file type the user wants to focus on for selection

    Returns:
    --------
    sel_file: str
        the selected path addition the user selected to be returned
    """

    files_found = [f for f in os.listdir(day_folder) if f.endswith(file_type)]
    if len(files_found) > 1:
        incorrect_data = True
        while incorrect_data:
            print(*files_found, sep='\n')
            sel_file = str(input(f'There are multiple {file_type[1:].capitalize()} Files Please Select One from Above: '))
            if sel_file in files_found:
                incorrect_data = False
            else:
                print(f'Not valid {file_type[1:].capitalize()} file')
    else:
        sel_file = files_found[0]
    return sel_file


def read_kwik_data(kwik_path, verbose=False):
    """ Read Spike Related information from the Kwik File

    Parameters:
    -----------
    kwik_path: file path #TODO: Switch to Pathlib
        path to the Kwik file
    verbose: bool
        if True prints status statement

    Returns:
    --------
    Spikedata: dict
        data needed to render spiking activity fromm the Kwik file

    Notes:
    ------
    Relevent Structure of KWIK Files (As I understand them 4/11/2018)
    Kwik:
        ├──*.kwik
        |    ├──'channel_groups'
        |    |    ├──[#] Relative channel index from 0 to shanksize-1 (May not be True with Zeke's Early Data z020, z007)
        |    |    |    ├──'spikes'
        |    |    |    |    ├──'time_samples': [N-long EArray of UInt64]
        |    |    |    |    |
        |    |    |    |    ├──'clusters'
        |    |    |    |    |   └──'main': [N-long EArray of UInt64]
        |    |
        |    ├──'recordings'
        |    |    ├──[#] # Recording index from 0 to Nrecordings-1
        |    |    |    └──'start_sample': Start time of the Recording

    Source: https://github.com/klusta-team/kwiklib/wiki/Kwik-format

    """

    # Create Dictionary for Data
    Spikedata = {}

    # Read in Kwik File
    kwik_file = h5py.File(kwik_path, 'r')

    # Get Recording Starts (Multiple are taken in a day)
    RC = np.zeros((len(kwik_file['recordings'].keys()), 1))     # Initialize Empty array of the size of the data

    for rec_num in range(len(kwik_file['recordings'].keys())):
        RC[rec_num, 0] = kwik_file['recordings'][str(rec_num)].attrs['start_sample']
    Spikedata['recordingStarts'] = RC  # Pass recording Starts to Spikedata

    # Get the Spike Times and Cluster Identity
    ChGroup = kwik_file['channel_groups']
    for ch in ChGroup.keys():
        Spikedata['time_samples'] = ChGroup[ch]['spikes']['time_samples']  # Spike Times
        Spikedata['clusters'] = ChGroup[ch]['spikes']['clusters']['main']  # Cluster Identity

    # Kwik file data completion statement
    if verbose:
        print('Kwik File has ', np.unique(Spikedata['clusters']).shape[0], ' Neurons and ',
              len(Spikedata['recordings'].keys()), ' Recordings')

    return Spikedata

def read_kwe_file(kwe_path, verbose=False):
    """ Iterates through the kwe file and gets the Motif Starts and which Recording they come from

    Parameters:
    -----------
    kwe_path:
        path to the KWE File
    verbose: bool
        If True, it prints status of function

    Returns:
    --------
    kwe_data: dict
        dictionary of the events in the KWE file
        Keys: #TODO: Update this and fix it in the Code

    """

    # Initialize Empty Dict
    kwe_data = {}

    if verbose:
        print('Getting KWE Data from ', kwe_path)

    # Read the KWE File
    kwe_file = h5py.File(kwe_path, 'r')

    # Get the Start Times and Recording Number
    kwe_data['MotifTS'] = kwe_file['event_types']['singing']['motiff_1']['time_samples']  # Start Time
    kwe_data['MotifRec'] = kwe_file['event_types']['singing']['motiff_1']['recording']    # Recording Number

    return kwe_data



def main():
    # This script writes LFP, Spike or Song data to numpy array and saves it in current directory

    # Folder where birds are
    experiment_folder_path = '/net/expData/birdSong/ss_data'

    # [1] Select Bird
    bird_id = ask_data_path(start_path=experiment_folder_path, focus="Bird")  # Ask User to Select Bird
    bird_folder_path = os.path.join(experiment_folder_path, bird_id)                # Folder for the bird

    # [2] Select Session
    session = ask_data_path(start_path=bird_folder_path, focus="Session")  # Ask User to Select Session
    dayFolder = os.path.join(bird_folder_path, session)                    # Folder for Session

    # [3] Select Kwik File and get its Data
    kwik_file = get_data_path(day_folder=dayFolder, file_type='.kwik')  # Ask User to select Kwik File
    kwik_file_path = os.path.join(dayFolder, kwik_file)                 # Get Path to Seleted Kwik File
    kwik_data = read_kwik_data(kwik_path=kwik_file_path, verbose=True)  # Make Dict of Data from Kwik File
    
    # [4] Select the Kwe file
    kwe = get_data_path(day_folder=dayFolder, file_type='.kwe')     # Select KWE File
    kwe_file_path = os.path.join(dayFolder, kwe)                    # Get Path to Selected KWE File
    kwe_data = read_kwe_file(kwe_path= kwe_file_path, verbose=False)   # Read KWE Data into Dict
    

    # [5] Select the Kwd file
    kwd = get_data_path(day_folder=dayFolder, file_type='.kwd')
    kwdFile = h5py.File(os.path.join(dayFolder, kwd), 'r')

    # Showing where data is coming from
    print('Getting Data from ', kwik_file) #TODO: Is this supposed to print the KWD File?


    incorrect_data = True
    while incorrect_data:
        print('Datasets: LFP, Spike, Song')
        data_type = str(input('Please Choose Dataset From Above: '))
        if data_type == 'LFP' or data_type == 'Song' or data_type == 'Spike':
            incorrect_data = False
        else:
            print('Not valid dataset')
            print('Datasets: LFP, Spike, Song')
            data_type = str(input('Please Choose Dataset From Above: '))
    # Set song parameters
    before_t = int(input('How much time before a motif do you want?(integer) '))
    after_t = int(input('How much time after a motif do you want?(integer) '))
    SongLengthMS = 500 + before_t+after_t
    SamplingRate = 30000 
    index = 0
    # Loop through all Motif time starts

    for Motif in range(kwe_data['MotifTS'].shape[0]):
         
        # Get start time for motif and recording start
        MotifStartTime = kwe_data['MotifTS'][Motif]
        Recording = kwe_data['MotifRec'][Motif]
        LFPaA = kwdFile['recordings'][str(Recording)]['data']
        MotifRecordingStart = kwik_data['recordingStarts'][kwe_data['MotifRec'][Motif]]
        
        # Copy over time samples and clusters
        SpikeDataTS = np.array(kwik_data['time_samples'])
        SpikeDataCI = np.array(kwik_data['clusters'])
        
        #Create spike data holder with neurons by song length size
        BinnedSpikes = np.zeros((np.unique(SpikeDataCI).shape[0],SongLengthMS))
        
        # Get all the unique cluster ID's, Some values are skipped
        ClusterUL = np.unique(SpikeDataCI)
        
        # Get Start Time and End Time in samples for the motif
        StartTime = int(MotifStartTime+MotifRecordingStart-before_t*30)
        EndTime = int(StartTime + SongLengthMS*30)
        StartTimeLFP = int(MotifStartTime-before_t*30)
        EndTimeLFP = int(StartTimeLFP + SongLengthMS*30)
        
        #Print out info about motif
        print('On Motif ', (Motif+1),'/',kwe_data['MotifTS'].shape[0], ' With Sample Start ',StartTime)
        
        # Get spikes that are between the start and end sample time stamps
        tempSpikes = SpikeDataTS[np.where(np.logical_and(StartTime<SpikeDataTS,SpikeDataTS<EndTime))]
        
        # Get cluster ID's for spikes between start and end time
        tempCI = SpikeDataCI[np.where(np.logical_and(StartTime<SpikeDataTS,SpikeDataTS<EndTime))]
        # Set that binned motif into larger data structure with key the motif number/name
        NumKWDCh = LFPaA.shape[1]
        
        #Loop through all the spikes that were between start and end time
        for I in range(tempSpikes.shape[0]):
            # Get the unique cluster ID
            tempClusterID = np.where(tempCI[I]==ClusterUL)
            # Get what bin the spike belongs to
            tempBinID = np.floor((tempSpikes[I]-StartTime)/(30))
            # Add 1 to the spike count for that bin and cluster
            BinnedSpikes[tempClusterID,tempBinID] = BinnedSpikes[tempClusterID,tempBinID] + 1 
        if index == 0:
            if data_type == 'Spike':
                MsSpikes = np.zeros((BinnedSpikes.shape[0],BinnedSpikes.shape[1],kwe_data['MotifTS'].shape[0]))
                MsSpikes[:,:,index] = BinnedSpikes
            elif data_type == 'Song':
                Song = np.zeros((LFPaA[StartTimeLFP:EndTimeLFP,NumKWDCh-1:NumKWDCh].shape[0],LFPaA[StartTimeLFP:EndTimeLFP,NumKWDCh-1:NumKWDCh].shape[1],kwe_data['MotifTS'].shape[0]))
                Song[:,:,index] = LFPaA[StartTimeLFP:EndTimeLFP,NumKWDCh-1:NumKWDCh]
            else:
                LFP = np.zeros((LFPaA[StartTimeLFP:EndTimeLFP,0:NumKWDCh-1].shape[0],LFPaA[StartTimeLFP:EndTimeLFP,0:NumKWDCh-1].shape[1],kwe_data['MotifTS'].shape[0]))
                LFP[:,:,index] = LFPaA[StartTimeLFP:EndTimeLFP,0:NumKWDCh-1]
        else:
            if data_type == 'Spike':
                
                MsSpikes[:, :, index] = BinnedSpikes
            elif data_type == 'Song':
                
                Song[:, :, index] = LFPaA[StartTimeLFP:EndTimeLFP,NumKWDCh-1:NumKWDCh]
            else:

                LFP[:, :, index] = LFPaA[StartTimeLFP:EndTimeLFP,0:NumKWDCh-1]
        index = index + 1
        
        
    if data_type == 'Spike':
        print('Saving Spike Data to','SpikeData'+bird_id+session+'.npy')
        np.save('SpikeData'+bird_id+session,MsSpikes)
    elif data_type == 'Song':
        print('Saving Song Data to','SongData'+bird_id+session+'.npy')
        np.save('SongData'+bird_id+session,Song)
    else:
        print('Saving LFP Data to','LFPData'+bird_id+session+'.npy')
        np.save('LFPData'+bird_id+session,LFP)


if __name__ == "__main__":
    main()
