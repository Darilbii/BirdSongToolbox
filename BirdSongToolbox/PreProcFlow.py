import numpy as np
import os
import h5py
import sys
import scipy
import scipy.io.wavfile
from scipy.signal import butter


# Reconsider the Handling of SN_L, Gp_L, and Gp in the Freq_Bin Commands
# Command for Initiallizing work space with Access to both: All the Data and Ephysflow Commands
def initiate_path():
    """
This Code is used to construct a path to the Data Folder using both the os and sys modules
please
    :return: Path to the Bird Song Data
    """
    experiment_folder = '/net/expData/birdSong/'
    ss_data_folder = os.path.join(experiment_folder, 'ss_data')      # Path to All Awake Bird Data
    sys.path.append(os.path.join(experiment_folder, 'ephysflow'))    # Appends the module created by Gentner Lab
    return ss_data_folder


def get_birds_data(Bird_Id=str, Session=str, ss_data_folder=str):
    """
This code is used to grab the data from the Awake Free Behaving Experiments done by Zeke and store them in a format that
works with the Python Environment
    :param Bird_Id: Specify the Specific Bird you are going to be looking at
    :param Session: Specify which Session you will be working with
    :param ss_data_folder: This Parameter is created by the initiate_path
    :return: Returns a List containing the Designated Experiments Results, and the Labels for its Motifs
    """
    bird_id = Bird_Id
    sess_name = Session

    kwd_file_folder = os.path.join(ss_data_folder, bird_id, sess_name)
    kwd_files = [f for f in os.listdir(kwd_file_folder) if f.endswith('.kwd')]
    assert (len(kwd_files) == 1)
    kwd_file = kwd_files[0]
    print(kwd_file)  # Sanity Check to Make Sure You are working with the Correct File

    # open the file in read mode
    kwd_file = h5py.File(os.path.join(kwd_file_folder, kwd_file), 'r')

    # Dynamic Members Size
    Num_Member = kwd_file.get('recordings')  # Test for making the For Loop for HD5 file dynamic
    Num_Members = Num_Member.keys()
    P = len(Num_Members)

    # Import Data from the .kwd File.

    Entire_trial = []
    File_loc = 'recordings/'
    k = ''
    j = 0

    # Isolate and Store Data into Numpy Array. Then Store Numpy Array into a List.

    for j in range(0, P):
        k = File_loc + str(j) + '/data'
        print(k)  # This is a Sanity Check to Ensure the Correct Data is accessed
        Epoch_data = np.array(kwd_file.get(k))
        Entire_trial.append(Epoch_data)
        j += 1

    # File Structure Part 2
    kwe_files = [f for f in os.listdir(kwd_file_folder) if f.endswith('.kwe')]
    assert (len(kwe_files) == 1)
    kwe_file = kwe_files[0]
    print(kwe_file)  # Sanity Check to Make Sure You are working with the Correct File

    # open the file in read mode
    kwe_file = h5py.File(os.path.join(kwd_file_folder, kwe_file), 'r')

    # Import Data from the .kwe File.

    # Store the Labels and Markers to Variables
    epoch_label = np.array(kwe_file.get('event_types/singing/motiff_1/recording'))
    print('Number of Motifs:', epoch_label.size)  # Good to Know/Sanity Check
    # print('')
    start_time = np.array(kwe_file.get('event_types/singing/motiff_1/time_samples'))
    print('Number of Start Times:', start_time.size)  # Sanity Check The Two Numbers should be equal

    assert (start_time.size == epoch_label.size)  # Check to Make Sure they are the same Length

    print('')
    print(epoch_label)
    print('')
    print(start_time)

    return Entire_trial, epoch_label, start_time


def clip_all_motifs(Entire_trial, Labels=np.ndarray, Starts=np.ndarray, song_length=str, Gaps=str):
    """
Command that Clips and Store Motifs or Bouts with a given Set of Parameters: Song Length, and Gap Length.

    :param Entire_trial:
    :param Labels:
    :param Starts:
    :param Song_Length:
    :param Gaps:
    :return:
    """
    All_Songs = []
    Motif_T = []
    Epoch_w_motif = []
    Testes = []

    Song_length = song_length  # Expected Song Duration in Seconds
    Gap = Gaps  # How much time before and after to add

    SN_L = int(Song_length * 30000)
    Gp = int(Gap * 30000)
    Gp_L = Gp * 2
    ############## SN_L and GP aren't integers which causes problems downstream, Changing this to int also causes problems

    fs = 30000.0  # 30 kHz
    lowcut = 400.0
    highcut = 10000.0

    # Motif_starts = []
    # New_Labels

    z = Labels.size
    stop_time = Starts + 30000 * Song_length
    i = 0

    for i in range(0, z):
        j = int(Labels[i])
        Holder = []
        Epoch_w_motif = Entire_trial[j]
        Motif_T = Epoch_w_motif[int(Starts[i] - Gp):int(stop_time[i] + Gp), :]
        #         Holder = scipy.signal.lfilter( bT, aT, Motif_T[:,16])
        Holder = butter_bandpass_filter(Motif_T[:, 16], lowcut, highcut, fs, order=2)
        Motif_T[:, 16] = Holder
        All_Songs.append(Motif_T[:, :])
    # All_Songs.append(Epoch_w_motif[int(start_time[i]-Gp):int(stop_time[i]+Gp),:])
    #         i += 1
    print('Song Motifs Acquired')
    return All_Songs, SN_L, Gp_L, Gp


# noinspection PyTupleAssignmentBalance
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  # Pycharm Freaks out here
    y = scipy.signal.filtfilt(b, a, data)
    return y
