import pickle

import numpy as np
import os
import h5py

# from BirdSongToolbox.config.settings import DATA_PATH
# TODO: Switch File Pathing to Pathlib
# TODO: Write Test Scripts for these Functions


def ask_data_path(start_path, focus: str):
    """ Asks the User to select data they wish to select for adding to a file path

    Parameters:
    -----------
    start_path:
        path to look for data selection options
    focus: str
        Identifier type the User needs to focus on for selection
    Returns:
    -------
    selection: str
        the selected path addition the user selected to be returned
    """

    escape_words = ['quit', 'exit', 'stop']

    options = os.listdir(start_path)
    print('\n')
    print(*options, sep='\n')
    selection = str(input(f'Please Select a {focus}:'))

    # Make sure that the selected value is available
    incorrect_data = True
    while incorrect_data:
        if selection.lower() in escape_words:
            return None
        elif selection not in options:
            print(f'\n {focus} not available \n')
            print(*options, sep='\n')
            selection = str(input(f'\n Now Please Select a {focus}: '))
        else:
            print(f'{focus} Selected')
            incorrect_data = False

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
            sel_file = str(
                input(f'There are multiple {file_type[1:].capitalize()} Files Please Select One from Above: '))
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
    kwik_data: dict
        data needed to render spiking activity fromm the Kwik file
        keys: 'recordingStarts', 'time_samples', 'clusters'

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
    kwik_data = {}

    # Read in Kwik File
    kwik_file = h5py.File(kwik_path, 'r')

    # Get Recording Starts (Multiple are taken in a day)
    RC = np.zeros((len(kwik_file['recordings'].keys()), 1))  # Initialize Empty array of the size of the data

    for rec_num in range(len(kwik_file['recordings'].keys())):
        RC[rec_num, 0] = kwik_file['recordings'][str(rec_num)].attrs['start_sample']
    kwik_data['recordingStarts'] = RC  # Pass recording Starts to kwik_data

    # Get the Spike Times and Cluster Identity
    ChGroup = kwik_file['channel_groups']
    for ch in ChGroup.keys():
        kwik_data['time_samples'] = ChGroup[ch]['spikes']['time_samples']  # Spike Times
        kwik_data['clusters'] = ChGroup[ch]['spikes']['clusters']['main']  # Cluster Identity

    # Kwik file data completion statement
    if verbose:
        print('Kwik File has ', np.unique(kwik_data['clusters']).shape[0], ' Neurons and ',
              len(kwik_file['recordings'].keys()), ' Recordings')

    return kwik_data


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
        Keys:
            'motif_st': [# of Motifs]
            'motif_rec_num': [# of Motifs]
    """

    # Initialize Empty Dict
    kwe_data = {}

    if verbose:
        print('Getting KWE Data from ', kwe_path)

    # Read the KWE File
    kwe_file = h5py.File(kwe_path, 'r')

    # Get the Start Times and Recording Number
    kwe_data['motif_st'] = kwe_file['event_types']['singing']['motiff_1']['time_samples']  # Start Time
    kwe_data['motif_rec_num'] = kwe_file['event_types']['singing']['motiff_1']['recording']  # Recording Number

    return kwe_data


def get_song_data(kwd_file, kwe_data, song_len_ms, before_t):
    """ Gets audio information from the KWD file

    Parameters:
    -----------
    kwd_file: h5py.File
        KWD file imported using h5py library
    kwe_data: dict
        dictionary of the events in the KWE file
        Keys:
            'motif_st': [# of Motifs]
            'motif_rec_num': [# of Motifs]
    song_len_ms: int
        Length of Time desired to be Grabbed for the Motif in ms
    before_t: int
        The amount of time (ms) before the  motif to start the data collection

    Returns:
    --------
    song: ndarray
        Multidimensional array of Raw Audio Pressure Signal
        (Motif Length in Samples  x  1  x  Num. of Motifs)

     Notes:
    ------
        The raw data are saved as signed 16-bit integers, in the range -32768 to 32767. They don’t have a unit. To
    convert to microvolts, just  multiply by 0.195. This scales the data into the range ±6.390 mV,
    with 0.195 µV resolution (Intan chips have a ±5 mV input range).

    """

    for motif in range(kwe_data['motif_st'].shape[0]):

        # Get start time for motif and recording start
        motif_start_time = kwe_data['motif_st'][motif]    # Start Time of Motif in its Specific Recording
        motif_rec_num = kwe_data['motif_rec_num'][motif]  # Recording Number Motif Occurs During
        # motif_rec_start = kwik_data['recordingStarts'][kwe_data['motif_rec_num'][Motif]]  # Start Sample of Recording
        kwd_rec_raw_data = kwd_file['recordings'][str(motif_rec_num)]['data']  # Raw Data for this Recording Number

        # Get Start Time and End Time in samples for the motif
        start_time_lfp = int(motif_start_time - before_t * 30)
        end_time_lfp = int(start_time_lfp + song_len_ms * 30)

        # Print out info about motif
        print('On Motif ', (motif + 1), '/', kwe_data['motif_st'].shape[0])

        num_kwd_ch = kwd_rec_raw_data.shape[1]

        if motif == 0:
            song = np.zeros((kwd_rec_raw_data[start_time_lfp:end_time_lfp, num_kwd_ch - 1:num_kwd_ch].shape[0],
                             kwd_rec_raw_data[start_time_lfp:end_time_lfp, num_kwd_ch - 1:num_kwd_ch].shape[1],
                             kwe_data['motif_st'].shape[0]))
            song[:, :, motif] = kwd_rec_raw_data[start_time_lfp:end_time_lfp, num_kwd_ch - 1:num_kwd_ch]
        else:
            song[:, :, motif] = kwd_rec_raw_data[start_time_lfp:end_time_lfp, num_kwd_ch - 1:num_kwd_ch]

    song = song * .195  # Convert to µVs
    return song


def get_lfp_data(kwd_file, kwe_data, song_len_ms, before_t):
    """ Gets Neural Information from the KWD File and converts it to µV

    Parameters:
    -----------
    kwd_file: h5py.File
        KWD file imported using h5py library
    kwe_data: dict
        dictionary of the events in the KWE file
        Keys:
            'motif_st': [# of Motifs]
            'motif_rec_num': [# of Motifs]
    song_len_ms: int
        Length of Time desired to be Grabbed for the Motif in ms
    before_t: int
        The amount of time (ms) before the  motif to start the data collection

    Returns:
    --------
    lfp: ndarray
        Multidimensional array of Neural Raw signal Recording
        (Motif Length in Samples  x  Num. of Channels  x  Num. of Motifs)

    Notes:
    ------
        The raw data are saved as signed 16-bit integers, in the range -32768 to 32767. They don’t have a unit. To
    convert to microvolts, just  multiply by 0.195. This scales the data into the range ±6.390 mV,
    with 0.195 µV resolution (Intan chips have a ±5 mV input range).

    """

    for motif in range(kwe_data['motif_st'].shape[0]):

        # Get start time for motif and recording start
        motif_start_time = kwe_data['motif_st'][motif]    # Start Time of Motif in its Specific Recording
        motif_rec_num = kwe_data['motif_rec_num'][motif]  # Recording Number Motif Occurs During
        # motif_rec_start = kwik_data['recordingStarts'][kwe_data['motif_rec_num'][Motif]]  # Start Sample of Recording
        kwd_rec_raw_data = kwd_file['recordings'][str(motif_rec_num)]['data']  # Raw Data for this Recording Number

        # Get Start Time and End Time in samples for the motif
        start_time_lfp = int(motif_start_time - before_t * 30)
        end_time_lfp = int(start_time_lfp + song_len_ms * 30)

        # Print out info about motif
        print('On Motif ', (motif + 1), '/', kwe_data['motif_st'].shape[0])

        num_kwd_ch = kwd_rec_raw_data.shape[1]

        if motif == 0:
            lfp = np.zeros((kwd_rec_raw_data[start_time_lfp:end_time_lfp, 0:num_kwd_ch - 1].shape[0],
                            kwd_rec_raw_data[start_time_lfp:end_time_lfp, 0:num_kwd_ch - 1].shape[1],
                            kwe_data['motif_st'].shape[0]))
            lfp[:, :, motif] = kwd_rec_raw_data[start_time_lfp:end_time_lfp, 0:num_kwd_ch - 1]
        else:
            lfp[:, :, motif] = kwd_rec_raw_data[start_time_lfp:end_time_lfp, 0:num_kwd_ch - 1]

    lfp = lfp * .195  # Convert to µVs
    return lfp


def get_spike_data(kwe_data, kwik_data, song_len_ms, before_t):
    """ Gets Spiking Information from the KWIK file and organize it into useful forms

    Parameters:
    -----------
    kwe_data: dict
        dictionary of the events in the KWE file
        Keys:
            'motif_st': [# of Motifs]
            'motif_rec_num': [# of Motifs]
    kwik_data: dict
        data needed to render spiking activity fromm the Kwik file
        keys: 'recordingStarts', 'time_samples', 'clusters'
    song_len_ms: int
        Length of Time desired to be Grabbed for the Motif in ms
    before_t: int
        The amount of time (ms) before the  motif to start the data collection

    Returns:
    --------
    spike_data: ndarray
        Binned Spike information for all Motifs
        (Number of Clusters  x  Length of Motif in ms  x  Num. of Motifs)
    spike_time_data: list
        Spike timing data related to each motif as directed based on the song_len_ms and before_t parameters
        [Num. of Motifs]-> {'Cluster Number'  :  [Spike Time (Variable Length)]}

    """
    spike_time_data = []

    # Loop through all Motif time starts

    for motif in range(kwe_data['motif_st'].shape[0]):

        # [1] Get Motif Time Information:  start time for motif and recording start
        motif_start_time = kwe_data['motif_st'][motif]  # Start Time of Motif in its Specific Recording
        motif_rec_start = kwik_data['recordingStarts'][kwe_data['motif_rec_num'][motif]]  # Start Sample of Recording

        # [2] Copy over time samples and clusters
        spike_data_ts = np.array(kwik_data['time_samples'])  # Copy Array of Spike Times
        spike_data_cid = np.array(kwik_data['clusters'])     # Copy Array of Cluster ID

        # [3] Create spike data holder (Num Neurons x Song Length size)
        binned_spikes = np.zeros((np.unique(spike_data_cid).shape[0], song_len_ms))

        # [4] Get all the unique cluster ID's, Some values are skipped
        cluster_ids = np.unique(spike_data_cid)

        # [5] Get Start Time and End Time in samples for the motif
        start_time = int(motif_start_time + motif_rec_start - before_t * 30)
        end_time = int(start_time + song_len_ms * 30)

        # Print out info about motif
        print('On Motif ', (motif + 1), '/', kwe_data['motif_st'].shape[0], ' With Sample Start ', start_time)

        # [6] Copy All Spiking Information for the Motif
        # Get spikes that are between the start and end sample time stamps
        spike_times_temp = spike_data_ts[np.where(np.logical_and(start_time < spike_data_ts, spike_data_ts < end_time))]

        # Get cluster ID's for spikes between start and end time
        cid_temp = spike_data_cid[np.where(np.logical_and(start_time < spike_data_ts, spike_data_ts < end_time))]

        # [7] Re-organize Spike Data into Useful Forms
        # Make Dictionary of Empty List with the Cluster IDs as Keys
        spike_dict = {}
        for key in cluster_ids:
            spike_dict[key] = []

        # Loop through all the spikes that were between start and end time
        for spike_time, cluster_identity in zip(spike_times_temp, cid_temp):

            temp_cluster_id = np.where(cluster_identity == cluster_ids)      # Get the unique cluster ID
            temp_time_bin_id = int(np.floor((spike_time- start_time) / 30))  # Get what time bin the spike belongs to

            # Add 1 to the spike count for that bin and cluster
            binned_spikes[temp_cluster_id, temp_time_bin_id] = binned_spikes[temp_cluster_id, temp_time_bin_id] + 1

            # Add the Spike Time to its List in the Dictionary of Unit's Activity
            spike_dict[cluster_identity].append(spike_time - start_time)

        spike_time_data.append(spike_dict)  # Save the Spike Time Dictionary to the List for All Motifs

        # [8] Add the Binned Spikes to the Array of Spikes for all Motifs
        if motif == 0:
            spike_data = np.zeros((binned_spikes.shape[0], binned_spikes.shape[1], kwe_data['motif_st'].shape[0]))
            spike_data[:, :, motif] = binned_spikes
        else:
            spike_data[:, :, motif] = binned_spikes

    return spike_data, spike_time_data


########################## WAS LAST WORKING HERE##################################
# TODO: I need to save the absolute start times for reference against the hand labels
# TODO: Work on the Raster Plot Function for test the spikes prior to pre-processing everything

def main():
    """
    Algorithm Steps:
    [1] Select the Bird
    [2] Select the Recording Day
    [3] Select the Kwik File and get its Data
    [4] Select the Kwe File and get its Data
    [5] Select the Kwd File
    [6] Define Parameters for Grabbing Data
    [7] Get Song Data
    [8] Get LFP Data
    [9] Get Spike Data

    """

    # This script writes LFP, Spike or Song data to numpy array and saves it in current directory

    # Folder where birds are
    experiment_folder_path = '/net/expData/birdSong/ss_data'

    # [1] Select Bird
    bird_id = ask_data_path(start_path=experiment_folder_path, focus="Bird")  # Ask User to Select Bird
    bird_folder_path = os.path.join(experiment_folder_path, bird_id)  # Folder for the bird

    # [2] Select Session
    session = ask_data_path(start_path=bird_folder_path, focus="Session")  # Ask User to Select Session
    dayFolder = os.path.join(bird_folder_path, session)                    # Folder for Session

    # [3] Select Kwik File and get its Data
    kwik_file = get_data_path(day_folder=dayFolder, file_type='.kwik')  # Ask User to select Kwik File
    kwik_file_path = os.path.join(dayFolder, kwik_file)                 # Get Path to Selected Kwik File
    kwik_data = read_kwik_data(kwik_path=kwik_file_path, verbose=True)  # Make Dict of Data from Kwik File

    # [4] Select the Kwe file
    kwe = get_data_path(day_folder=dayFolder, file_type='.kwe')      # Select KWE File
    kwe_file_path = os.path.join(dayFolder, kwe)                     # Get Path to Selected KWE File
    kwe_data = read_kwe_file(kwe_path=kwe_file_path, verbose=False)  # Read KWE Data into Dict

    # [5] Select the Kwd file
    kwd = get_data_path(day_folder=dayFolder, file_type='.kwd')   # Select Kwd File
    kwd_file = h5py.File(os.path.join(dayFolder, kwd), 'r')        # Read Kwd Data into Dict

    # Showing where data is coming from
    print('Getting Data from ', kwd)

    # [6] Define Parameters for Grabbing Data
    # Select What Data type(s) to Get
    incorrect_data = True
    data_types = ['Datasets:', 'All', 'LFP', 'Spike', 'Song']  # Lists of Data Types available
    # print(*data_types, sep='\n')
    print('Datasets: LFP, Spike, Song, All')
    data_type = str(input('Please Choose Dataset From Above: '))
    while incorrect_data:
        if data_type in data_types:
            incorrect_data = False
        else:
            print('\n Not a valid Dataset')
            print(*data_types, sep='\n')
            data_type = str(input('Please Choose Dataset From Above: '))

    # Set Parameters for Grabbed Data
    before_t = int(input('How much time (ms) Before a motif do you want? (integer): '))
    after_t = int(input('How much time (ms) After a motif do you want? (integer): '))
    motif_dur = int(input("How long does this Bird's Motif last in milliseconds? (integer): "))
    song_len_ms = motif_dur + before_t + after_t  # Calculate the Duration of the Grabbed Data
    SamplingRate = 30000

    # TODO: Work Out where I will Save these intermediate data
    temp_destination = '/home/debrown/data/'

    # [7] Get Song Data
    if data_type == 'Song' or data_type == 'All':
        # Get Song Data
        song_data = get_song_data(kwd_file=kwd_file, kwe_data=kwe_data, song_len_ms=song_len_ms, before_t=before_t)

        # Save Song Data
        song_data_file = temp_destination + 'SongData' + '_' + bird_id + '_' + session
        print('Saving Song Data to', song_data_file + '.npy')
        np.save(song_data_file, song_data)

    # [8] Get LFP Data
    if data_type == 'LFP' or data_type == 'All':
        # Get LFP Data
        lfp_data = get_lfp_data(kwd_file=kwd_file, kwe_data=kwe_data, song_len_ms=song_len_ms, before_t=before_t)

        # Save LFP Data
        lfp_data_file = temp_destination + 'LFPData' + '_' + bird_id + '_' + session
        print('Saving LFP Data to', lfp_data_file + '.npy')
        np.save(lfp_data_file, lfp_data)

    # [9] Get Spike Data
    if data_type == 'Spike' or data_type == 'All':
        # Get Spike Data
        binned_spikes, spike_time_data = get_spike_data(kwe_data=kwe_data, kwik_data=kwik_data, song_len_ms=song_len_ms,
                                                        before_t=before_t)
        # Save Binned Spike Data
        bin_spike_data_file = temp_destination + 'SpikeData' + '_' + bird_id + '_' + session
        print('Saving Binned Spike Data to', bin_spike_data_file + '.npy')
        np.save(bin_spike_data_file, binned_spikes)

        # Save Spike Time Data
        file_name = 'SpikeTimeData' + '_' + bird_id + '_' + session + '.pckl'
        destination = temp_destination + file_name
        file_object = open(destination, 'wb')
        pickle.dump(spike_time_data, file_object)
        file_object.close()
        print('Saving Spike Time Data to', destination)


if __name__ == "__main__":
    main()
