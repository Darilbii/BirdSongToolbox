import numpy as np
import os 
import h5py


def inquire_data_path(start_path, focus: str):
    """ Asks the User to select data they wish to select for adding to a file path

    Parameters:
    -----------
    start_path:
        path to look for data selection options
    focus: str

    Returns:
    -------
    selection: str

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


def main():
    # This script writes LFP,Spike or Song data to numpy array and saves it in current directory

    # Folder where birds are
    EFolder = '/net/expData/birdSong/ss_data'

    # [1] Select Bird
    # Ask User to Select Bird
    BirdId = inquire_data_path(start_path=EFolder, focus="Bird")

    # Folder for the bird
    birdFolder = os.path.join(EFolder,BirdId)

    # [2] Select Session
    # Ask User to Select Session
    Session = inquire_data_path(start_path=birdFolder, focus="Session")

    # Folder for Session
    dayFolder = os.path.join(birdFolder, Session)
################### Last HERE ######################################
    Spikedata = {}
    kweD = {}
    kwik_files = [f for f in os.listdir(dayFolder) if f.endswith('.kwik')]
    if len(kwik_files) > 1:
        incorrectData = True
        while incorrectData:
            print(kwik_files)
            kwik_file = str(input('There are multiple Kwik Files Please Select One from Above: '))
            if kwik_file in kwik_files:
                incorrectData = False
            else:
                print('Not valid kwik file')
    else:
        kwik_file = kwik_files[0]
                
    # Read in Kwik File
    BirdFile = h5py.File(os.path.join(dayFolder, kwik_file), 'r')
    
    # Get Recording Starts
    RC = np.zeros((len(BirdFile['recordings'].keys()),1))
    for r in range(len(BirdFile['recordings'].keys())):
        RC[r,0] = BirdFile['recordings'][str(r)].attrs['start_sample']
    # Pass recording Starts to Spikedata
    Spikedata['recordingStarts'] = RC
    
    # Go through all channels and get time samples of spikes and the cluster ID
    ChGroup = BirdFile['channel_groups']
    for ch in ChGroup.keys():
        temp = ChGroup[ch]
        temp2 = temp['spikes']
        Spikedata['time_samples'] = temp2['time_samples']
        Spikedata['clusters'] = temp2['clusters']['main']
    
    # Kwik file data completion statement
    print('Kwik File has ',np.unique(Spikedata['clusters']).shape[0], ' Neurons and ', len(BirdFile['recordings'].keys()), ' Recordings')
    
    # Go through all the kwe files and get Motif Starts and what recording they come from
    kwe_files = [f for f in os.listdir(dayFolder) if f.endswith('.kwe')]
    if len(kwe_files) > 1:
        incorrectData = True
        while incorrectData:
            print(kwe_files)
            kwe = str(input('There are multiple Kwe Files Please Select One from Above: '))
            if kwe in kwe_files:
                incorrectData = False
            else:
                print('Not valid kwe file')
    else:
        kwe = kwe_files[0]
   
    kweFile = h5py.File(os.path.join(dayFolder,kwe),'r')
    print('Getting KWE Data from ',kwe)
    kweD['MotifTS'] = kweFile['event_types']['singing']['motiff_1']['time_samples']
    kweD['MotifRec'] = kweFile['event_types']['singing']['motiff_1']['recording']
    

    kwd_files = [k for k in os.listdir(dayFolder) if k.endswith('.kwd')]
    if len(kwd_files) > 1:
        incorrectData = True
        while incorrectData:
            print(kwd_files)
            kwd = str(input('There are multiple Kwd Files Please Select One from Above: '))
            if kwd in kwd_files:
                incorrectData = False
            else:
                print('Not valid kwd file')
    else:
        kwd = kwd_files[0]
    kwdFile = h5py.File(os.path.join(dayFolder,kwd),'r')

        
    # Showing where data is coming from
    print('Getting Data from ',kwik_file)


    incorrectData = True
    while incorrectData:
        print('Datasets: LFP, Spike, Song')
        dataType = str(input('Please Choose Dataset From Above: '))
        if dataType == 'LFP' or dataType == 'Song' or dataType == 'Spike':
            incorrectData = False
        else:
            print('Not valid dataset')
            print('Datasets: LFP, Spike, Song')
            dataType = str(input('Please Choose Dataset From Above: '))
    # Set song parameters
    BeforeT = int(input('How much time before a motif do you want?(integer) '))
    AfterT = int(input('How much time after a motif do you want?(integer) '))
    SongLengthMS = 500 + BeforeT+AfterT
    SamplingRate = 30000 
    index = 0
    # Loop through all Motif time starts
    for Motif in range(kweD['MotifTS'].shape[0]):
         
        # Get start time for motif and recording start
        MotifStartTime = kweD['MotifTS'][Motif]
        Recording = kweD['MotifRec'][Motif]
        LFPaA = kwdFile['recordings'][str(Recording)]['data']
        MotifRecordingStart = Spikedata['recordingStarts'][kweD['MotifRec'][Motif]]
        
        # Copy over time samples and clusters
        SpikeDataTS = np.array(Spikedata['time_samples'])
        SpikeDataCI = np.array(Spikedata['clusters'])
        
        #Create spike data holder with neurons by song length size
        BinnedSpikes = np.zeros((np.unique(SpikeDataCI).shape[0],SongLengthMS))
        
        # Get all the unique cluster ID's, Some values are skipped
        ClusterUL = np.unique(SpikeDataCI)
        
        # Get Start Time and End Time in samples for the motif
        StartTime = int(MotifStartTime+MotifRecordingStart-BeforeT*30)
        EndTime = int(StartTime + SongLengthMS*30)
        StartTimeLFP = int(MotifStartTime-BeforeT*30)
        EndTimeLFP = int(StartTimeLFP + SongLengthMS*30)
        
        #Print out info about motif
        print('On Motif ', (Motif+1),'/',kweD['MotifTS'].shape[0], ' With Sample Start ',StartTime)
        
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
            if dataType == 'Spike':
                MsSpikes = np.zeros((BinnedSpikes.shape[0],BinnedSpikes.shape[1],kweD['MotifTS'].shape[0]))
                MsSpikes[:,:,index] = BinnedSpikes
            elif dataType == 'Song':
                Song = np.zeros((LFPaA[StartTimeLFP:EndTimeLFP,NumKWDCh-1:NumKWDCh].shape[0],LFPaA[StartTimeLFP:EndTimeLFP,NumKWDCh-1:NumKWDCh].shape[1],kweD['MotifTS'].shape[0]))
                Song[:,:,index] = LFPaA[StartTimeLFP:EndTimeLFP,NumKWDCh-1:NumKWDCh]
            else:
                LFP = np.zeros((LFPaA[StartTimeLFP:EndTimeLFP,0:NumKWDCh-1].shape[0],LFPaA[StartTimeLFP:EndTimeLFP,0:NumKWDCh-1].shape[1],kweD['MotifTS'].shape[0]))
                LFP[:,:,index] = LFPaA[StartTimeLFP:EndTimeLFP,0:NumKWDCh-1]
        else:
            if dataType == 'Spike':
                
                MsSpikes[:,:,index] = BinnedSpikes
            elif dataType == 'Song':
                
                Song[:,:,index] = LFPaA[StartTimeLFP:EndTimeLFP,NumKWDCh-1:NumKWDCh]
            else:

                LFP[:,:,index] = LFPaA[StartTimeLFP:EndTimeLFP,0:NumKWDCh-1]
        index = index + 1
        
        
    if dataType == 'Spike':
        print('Saving Spike Data to','SpikeData'+BirdId+Session+'.npy')
        np.save('SpikeData'+BirdId+Session,MsSpikes)
    elif dataType == 'Song':
        print('Saving Song Data to','SongData'+BirdId+Session+'.npy')
        np.save('SongData'+BirdId+Session,Song)
    else:
        print('Saving LFP Data to','LFPData'+BirdId+Session+'.npy')
        np.save('LFPData'+BirdId+Session,LFP)


if __name__ == "__main__":
    main()
