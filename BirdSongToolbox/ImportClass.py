import numpy as np
import os
import scipy.io as sio
from pathlib import Path
# import BirdSongToolbox.config.settings as settings
from .config.settings import DATA_PATH

# TODO: Convert Pathing to be through Pathlib and Centralized like Songbird-Repository

# Class function for Importing PrePd Data
### Confirm this works as intended
class Import_PrePd_Data():
    """Import Prepared (PrePd) Data and its accompanying meta-data into the workspace for analysis

    Note: This data has been prepared using self created Matlab scripts that required hand labeling.
    Make sure that you have used the correct Data Preping script.

    Naming Convention:
    ------------------
    <bird_id>_day_<#>

    Methods:
    --------
    Describe(self): Prints Relevant information about the Imported Data

    Parameters:
    -----------
    bird_id: str
        Bird Indentifier to Locate Specified Bird's data folder
    sess_name: str
        Experiment Day to Locate it's Folder
    data_type: string
        String Directing the Type of Neural Signal to Import, (Options: 'LPF_DS', 'LPF', 'Raw')

    Objects:
    --------
    .bird_id: str
        Bird Indentifier to Locate Specified Bird's data folder
    .date: str
        Experiment Day to Locate it's Folder
    .data_type = str
        Description of the Type of Data being Imported
        Options:
            {'LPF': Low Pass Filter, 'LPF_DS': Low Pass Filtered & Downsampled, 'raw': Raw Data}
    .Sn_Len = int
        Time Duration of Birds Motif (in Samples)
    .Gap_Len = int
        Duration of Buffer used for Trials (in Samples)
    .Fs = int
        Sample Frequency of Data (in Samples)
    .Num_Chan = int
        Number of Recording Channels used on Bird
    .Bad_Channels = list
        List of Channels with Noise to be excluded from Common Average Referencing
    .Song_Neural: list
        User Designated Neural data during Song Trials
        [Number of Trials]-> [Trial Length (Samples @ User Designated Sample Rate) x Ch]
    .Silence_Neural: list
        User Designated Neural Data during Silent Trials
        [Number of Trials]-> [Trial Length (Samples @ User Designated Sample Rate) x Ch]
    .Song_Audio: list
        Audio of Trials, centered on motif
        [Number of Trials]-> [Trial Length (Samples @ 30KHz) x 1]
    .Num_Motifs: int
        Number of Motifs in data set
    .Silence_Audio: list
        Audio of Silents Trials
        [Number of Trials]-> [Trial Length (Samples @ 30KHz) x 1]
    .Num_Silence: int
        Number of Examples of Silence
    .Song_Quality: list
        Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
        [Number of Trials x 1 (numpy.unicode_)]
    .Song_Locations: list [Number of Trials x 1 (numpy.unicode_)]
        Describes the Location of the Motif in the BOut, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
        [Number of Trials x 1 (numpy.unicode_)]
    .Song_Syl_Drop: list
        Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
        [Number of Trials x 1 (numpy.unicode_)]
        *** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility***
    .Good_Motifs: np.ndarray
        Index of All Good Motifs, 'Good' is defined as having little noise and no dropped (or missing) syllables
    .First_Motifs: np.ndarray
        Index of All Good First Motifs, this motif is the first motif in a bout and is classified as 'Good'
    .Last_Motifs: np.ndarray
        Index of All Good Last Motifs, this motif is the last motif in a bout and is classified as 'Good'
    .Bad_Motifs: np.ndarray
        Index of All Bad Motifs with no dropped syllables, These motifs have interferring audio noise
    .LS_Drop: np.ndarray
        Index of All Bad Motifs with the last syllable dropped, These motifs are classified as Bad
    .All_First_Motifs: np.ndarray
        Index of All First Motifs in a Bout Regardless of Quality label, This is Useful for Clip-wise (Series) Analysis
    .All_Last_Motifs: np.ndarray
        Index of All Last Motifs in a Bout Regardless of Quality label, This is Useful for Clip-wise (Series) Analysis
    .Good_Mid_Motifs: np.ndarray
        Index of All Good Motifs in the middle of a Bout Regardless of Quality label, This is Useful for Clip-wise (Series)
        Analysis

    Example:
    --------
    >>>
    """

    def __init__(self, bird_id, sess_name, data_type='LPF_DS', location=None):
        """Entire class self-constructs using modularized functions from Import_Birds_PrePd_Data() Use as a referenc to debug

        Parameters:
        -----------
            bird_id: str
                Bird Indentifier to Locate Specified Bird's data folder
            sess_name: str
                Experiment Day to Locate it's Folder
            data_type: string
                String Directing the Type of Neural Signal to Import, (Options: 'LPF_DS', 'LPF', 'Raw')
            location: str or Path object, (Optional)
                Location to search for the data other than default DATA_PATH (Optional)
        """
        assert type(bird_id) == str
        assert type(sess_name) == str

        # Initialize Session Identifiers
        self.bird_id = bird_id
        self.date = sess_name
        self.data_type = data_type

        # Basic Setup for path Creation
        # experiment_folder = '/net/expData/birdSong/'
        # experiment_folder = settings.DATA_PATH
        if location is None:
            experiment_folder = DATA_PATH
        else:
            experiment_folder = location

        if isinstance(experiment_folder, str):
            experiment_folder = Path(experiment_folder)
            assert experiment_folder.exists(), "Directory pointed by location parameter does not exist"
            experiment_folder.resolve()

        # prepd_ss_data_folder = os.path.join(experiment_folder, 'ss_data_Processed')
        prepd_ss_data_folder = experiment_folder / 'ss_data_Processed'

        # Modularized Data Import Steps
        self.Identify_Bird()  ## Determine Data's Metadata (Predefined Options based on Annotation)

        self._ImportSwitch(prepd_ss_data_folder)  ## Import User Designated Neural Data for Song and Silence
        self.Get_Song_Audio(prepd_ss_data_folder)  ## Song: Store the Filtered Audio Data
        self.Get_Silence_Audio(prepd_ss_data_folder)  ## Silence: Store the Filtered Audio Data
        self.Get_Hand_Labels(prepd_ss_data_folder)  ## Store the Different Types of Hand Labels into Seperate Lists

        # Modularized Indexes of Relevant Handlabel Pairs
        self.Locate_All_Good_Motifs()
        self.Locate_Good_First_Motifs()
        self.Locate_Bad_Full_Motifs()
        self.Locate_Good_Last_Motifs()
        self.Locate_Last_Syll_Dropped()
        self.Locate_Bouts()
        self.Locate_All_Last_Motifs()
        self.Locate_Good_Mid_Motifs()

        # Confirm completion of Import to User
        self.Describe()

    def Identify_Bird(self):
        """Acquire Standard Descriptive Information on Data based on Bird Identity and User Instructions

        This Function Must Be Maintained and Updated as More Birds are added
        * Must Create a Table of Birds and Relevant Information on them *
        """
        # Consider Moving this to be a Global Variable of this Package
        Song_Length = {'z020': .5, 'z007': .8, 'z017': .8}  # Dictionary of Bird's Specific Motif Length (Seconds)
        Gap_Length = {'z020': 4, 'z007': 4, 'z017': 4}  # Dictionary of Bird's Specific Time Buffer (Seconds)
        Channel_Count = {'z020': 16, 'z007': 32, 'z017': 16}  # Dictionary of Bird's Probe's Channel Count
        Bad_Channels = {'z020': [2], 'z007': [], 'z017': []}  # Dictionary of Bird's Bad Channels ***** Maybe Overkill
        Sample_Frequency = {'Raw': 30000, 'LPF': 30000,
                            'LPF_DS': 1000, }  # Dictionary of Possible Sample Frequencies (Samples per Second)
        print(
            '* Must Create a Table of Birds and Relevant Information on them *')  # To Make Sure I Return to This Idea in the Future

        # Validate Input is correct
        assert (self.bird_id in Song_Length.keys()) == True, 'The Bird: %s is not valid for Song_Length' % self.bird_id
        assert (self.bird_id in Gap_Length.keys()) == True, 'The Bird: %s is not valid for Gap_Length' % self.bird_id
        assert (
                       self.bird_id in Channel_Count.keys()) == True, 'The Bird: %s is not valid for Channel_Count' % self.bird_id
        assert (
                       self.bird_id in Bad_Channels.keys()) == True, 'The Bird: %s is not valid for Bad_Channels' % self.bird_id
        assert (
                       self.data_type in Sample_Frequency.keys()) == True, 'That Name: %s is not valid for Data type' % self.data_type

        # Actual Action Step: Store Recording's Metadata
        self.Sn_Len = int(Song_Length[self.bird_id] * Sample_Frequency[self.data_type])  # Determine Length in Samples
        self.Gap_Len = int(Gap_Length[self.bird_id] * Sample_Frequency[self.data_type])  # Determine Length in Samples
        self.Num_Chan = Channel_Count[self.bird_id]  # Get Channel Count
        self.Bad_Channels = Bad_Channels[self.bird_id]  # Get Bad Channels
        self.Fs = Sample_Frequency[self.data_type]

    def _ImportSwitch(self, Prepd_ss_data_folder):
        """Functional Switch to Control what type of Neural Data is imported"""

        if self.data_type == 'LPF_DS':
            self.Song_Neural = self.Get_LPF_DS_Song(Prepd_ss_data_folder)
            self.Silence_Neural = self.Get_LPF_DS_Silence(Prepd_ss_data_folder)
        elif self.data_type == 'Raw':
            self.Song_Neural = self.Get_Raw_Song(Prepd_ss_data_folder)
            self.Silence_Neural = self.Get_Raw_Silence(Prepd_ss_data_folder)
        elif self.data_type == 'LPF':
            self.Song_Neural = self.Get_LPF_Song(Prepd_ss_data_folder)
            self.Silence_Neural = self.Get_LPF_Silence(Prepd_ss_data_folder)
        else:
            print('Invalid Neural Data Type')

    def Get_LPF_DS_Song(self, Prepd_ss_data_folder):
        """Song: Store the Low Pass Filtered & Downsampled Neural Data
        Parameters:
        -----------
            Song_File: str
                path to data
        """
        # Song_File = os.path.join(Prepd_ss_data_folder, self.bird_id, self.date, 'Song_LFP_DS.mat')
        Song_File = Prepd_ss_data_folder / self.bird_id / self.date / 'Song_LFP_DS.mat'
        Song_LPF_DS_Data = []
        Mat_File = sio.loadmat(Song_File)
        Mat_File_Filt = Mat_File['Song_LFP_DS']
        Numb_Motifs = len(Mat_File_Filt)

        for i in range(0, Numb_Motifs):
            Song_LPF_DS_Data.append(np.transpose(Mat_File_Filt[i, 0]))

        self.Num_Motifs = Numb_Motifs

        return Song_LPF_DS_Data

    def Get_LPF_Song(self, Prepd_ss_data_folder):
        """Song: Store the Low Pass Filtered & Downsampled Neural Data
        Parameters:
        -----------
            Song_File: str
                path to data
        """
        # Song_File = os.path.join(Prepd_ss_data_folder, self.bird_id, self.date, 'Song_LFP.mat')
        Song_File = Prepd_ss_data_folder / self.bird_id / self.date / 'Song_LFP.mat'
        Song_LPF_Data = []
        Mat_File = sio.loadmat(Song_File)
        Mat_File_Filt = Mat_File['Song_LFP']
        Numb_Motifs = len(Mat_File_Filt)

        for i in range(0, Numb_Motifs):
            Song_LPF_Data.append(np.transpose(Mat_File_Filt[i, 0]))

        self.Num_Motifs = Numb_Motifs

        return Song_LPF_Data

    def Get_Raw_Song(self, Prepd_ss_data_folder):
        """Song: Store the Raw Neural Data
        Parameters:
        -----------
            Song_File: str
                path to data
        """
        # Song_File = os.path.join(Prepd_ss_data_folder, self.bird_id, self.date, 'Song_Raw.mat')
        Song_File = Prepd_ss_data_folder / self.bird_id / self.date / 'Song_Raw.mat'
        Song_Raw_Data = []
        Mat_File = sio.loadmat(Song_File)
        Mat_File_Filt = Mat_File['Song_Raw']
        Numb_Motifs = len(Mat_File_Filt)

        for i in range(0, Numb_Motifs):
            Song_Raw_Data.append(np.transpose(Mat_File_Filt[i, 0]))
        self.Num_Motifs = Numb_Motifs

        return Song_Raw_Data

    def Get_Song_Audio(self, Prepd_ss_data_folder):
        """Song: Store the Filtered Audio Data"""
        # Song_File = os.path.join(Prepd_ss_data_folder, self.bird_id, self.date, 'Song_Audio.mat')
        Song_File = Prepd_ss_data_folder / self.bird_id / self.date / 'Song_Audio.mat'

        Song_Audio_Data = []
        Mat_File = sio.loadmat(Song_File)
        Mat_File_Filt = Mat_File['Song_Audio']

        Song_Audio_Data = []
        for i in range(0, self.Num_Motifs):
            Song_Audio_Data.append(np.transpose(Mat_File_Filt[i, 0]))
        self.Song_Audio = Song_Audio_Data

    def Get_LPF_DS_Silence(self, Prepd_ss_data_folder):
        """Silence: Store the Low Pass Filtered & Downsampled Neural Data"""

        # Silence_File = os.path.join(Prepd_ss_data_folder, self.bird_id, self.date, 'Silence_LFP_DS.mat')
        Silence_File = Prepd_ss_data_folder / self.bird_id / self.date / 'Silence_LFP_DS.mat'
        Silence_LPF_DS_Data = []
        Mat_File = sio.loadmat(Silence_File)
        Mat_File_Filt = Mat_File['Silence_LFP_DS']
        Numb_Sil_Ex = len(Mat_File_Filt)

        for i in range(0, Numb_Sil_Ex):
            Silence_LPF_DS_Data.append(np.transpose(Mat_File_Filt[i, 0]))

        self.Num_Silence = Numb_Sil_Ex
        return Silence_LPF_DS_Data

    def Get_LPF_Silence(self, Prepd_ss_data_folder):
        """Silence: Store the Low Pass Filtered & Downsampled Neural Data"""

        # Silence_File = os.path.join(Prepd_ss_data_folder, self.bird_id, self.date, 'Silence_LFP.mat')
        Silence_File = Prepd_ss_data_folder / self.bird_id / self.date / 'Silence_LFP.mat'

        Silence_LPF_Data = []
        Mat_File = sio.loadmat(Silence_File)
        Mat_File_Filt = Mat_File['Silence_LFP']
        Numb_Sil_Ex = len(Mat_File_Filt)

        for i in range(0, Numb_Sil_Ex):
            Silence_LPF_Data.append(np.transpose(Mat_File_Filt[i, 0]))

        self.Num_Silence = Numb_Sil_Ex
        return Silence_LPF_Data

    def Get_Raw_Silence(self, Prepd_ss_data_folder):
        """Silence: Store the Raw Neural Data
        Parameters:
        -----------
            Song_File: str
                path to data
        """
        # Song_File = os.path.join(Prepd_ss_data_folder, self.bird_id, self.date, 'Silence_Raw.mat')
        Song_File = Prepd_ss_data_folder / self.bird_id / self.date / 'Silence_Raw.mat'

        Silence_Raw_Data = []
        Mat_File = sio.loadmat(Song_File);
        Mat_File_Filt = Mat_File['Silence_Raw'];
        Numb_Sil_Ex = len(Mat_File_Filt)

        for i in range(0, Numb_Sil_Ex):
            Silence_Raw_Data.append(np.transpose(Mat_File_Filt[i, 0]))

        self.Num_Silence = Numb_Sil_Ex
        return Silence_Raw_Data

    def Get_Silence_Audio(self, Prepd_ss_data_folder):
        """Silence: Store the Filtered Audio Data"""

        # Silence_File = os.path.join(Prepd_ss_data_folder, self.bird_id, self.date, 'Silence_Audio.mat')
        Silence_File = Prepd_ss_data_folder / self.bird_id / self.date / 'Silence_Audio.mat'

        Silence_Audio_Data = []
        Mat_File = sio.loadmat(Silence_File);
        Mat_File_Filt = Mat_File['Silence_Audio'];

        Silence_Audio_Data = []
        for i in range(0, self.Num_Silence):
            Silence_Audio_Data.append(np.transpose(Mat_File_Filt[i, 0]))
        self.Silence_Audio = Silence_Audio_Data

    def Get_Hand_Labels(self, Prepd_ss_data_folder):
        """Stores the Different Types of Labels into Seperate Lists"""

        # Labels_File = os.path.join(Prepd_ss_data_folder, self.bird_id, self.date, 'Labels_py.mat')
        Labels_File = Prepd_ss_data_folder / self.bird_id / self.date / 'Labels_py.mat'

        Labels_Quality = []
        Labels_Location = []
        Labels_Syl_Drop = []

        Mat_File = sio.loadmat(Labels_File);
        Mat_File_Filt = Mat_File['Motif_Labels'];
        Numb_Motifs = len(Mat_File_Filt);

        assert self.Num_Motifs == Numb_Motifs

        # Store the Low Pass Filtered & Downsampled Neural Data
        for i in range(0, self.Num_Motifs):
            Labels_Quality.append(np.transpose(Mat_File_Filt[i, 0]))
            Labels_Location.append(np.transpose(Mat_File_Filt[i, 1]))
            Labels_Syl_Drop.append(np.transpose(Mat_File_Filt[i, 2]))

        self.Song_Quality = Labels_Quality
        self.Song_Locations = Labels_Location
        self.Song_Syl_Drop = Labels_Syl_Drop

    def Locate_All_Good_Motifs(self):
        """Create Index for All Good Motifs

        Parameters:
        -----------
        .Song_Quality: list
            Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Locations: list
            Describes the Location of the Motif in the Bout, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Syl_Droplist
            Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
            [Number of Trials x 1 (numpy.unicode_)]
            ** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility**

        Returns:
        --------
        .Good_Motifs: list
            Index of All Good Motifs, 'Good' is defined as having little noise and no dropped (or missing) syllables
        """

        # 1. All Good Motifs
        # 1.1 Initialize Variables and Memory
        Quality_Holder = np.zeros(len(self.Song_Quality))  # Allocate Memory Equal to Number of Motifs for Indexing

        # 1.2 Fill Logical Index for finding Good
        for i in range(len(self.Song_Quality)):
            if self.Song_Quality[i][0] == 'Good':  # Locate Each Good Label
                Quality_Holder[i] = 1  # Create Index of Selected Label

        Good_Motifz = np.where(Quality_Holder == 1)  # Make Index for Where it is True
        Good_Motifz = Good_Motifz[0]  # Weird Needed Step
        self.Good_Motifs = Good_Motifz

    def Locate_Good_First_Motifs(self):
        """Create Index for All Good First Motifs

        Parameters:
        -----------
        .Song_Quality: list
            Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Locations: list
            Describes the Location of the Motif in the Bout, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Syl_Droplist
            Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
            [Number of Trials x 1 (numpy.unicode_)]
            ** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility**


        Returns:
        --------
        .First_Motifs: list
            Index of All Good First Motifs, this motif is the first motif in a bout and is classified as 'Good'
        """
        # 2. Good First Motifs
        # 2.1 Initialize Variables and Memory
        assert len(self.Song_Locations) == len(self.Song_Quality)
        First_Holder = np.zeros(len(self.Song_Locations))  # Allocate Memory size of Number of Motifs for Indexing

        # 2.2 Fill Logical for Good First Motifs
        for i in range(len(self.Song_Quality)):
            if self.Song_Quality[i][0] == 'Good':
                if self.Song_Locations[i][0] == 'Beginning':  # Locate Desired Label Combination
                    First_Holder[i] = 1  # Mark them
        First_Motifz = np.where(First_Holder == 1)  # Create Index of Selected Label
        First_Motifz = First_Motifz[0]  # Weird Needed Step
        self.First_Motifs = First_Motifz

    def Locate_Good_Last_Motifs(self):
        """Create Index for All Good Last Motifs

        Parameters:
        -----------
        .Song_Quality: list
            Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Locations: list
            Describes the Location of the Motif in the Bout, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Syl_Droplist
            Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
            [Number of Trials x 1 (numpy.unicode_)]
            ** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility**


        Returns:
        --------
        .Last_Motifs: list
            Index of All Good Last Motifs, this motif is the last motif in a bout and is classified as 'Good'
        """

        # 3. Good Last Motifs
        # 3.1 Initialize Variables and Memory
        assert len(self.Song_Locations) == len(self.Song_Quality)
        Last_Holder = np.zeros(len(self.Song_Locations))  # Allocate Memory for Indexing

        # 3.2 Fill Logical for Good First Motifs
        for i in range(len(self.Song_Quality)):
            if self.Song_Quality[i][0] == 'Good':
                if self.Song_Locations[i][0] == 'Ending':  # Locate Desired Label Combination
                    Last_Holder[i] = 1  # Mark them
        Last_Motifz = np.where(Last_Holder == 1)  # Create Index of Selected Label
        Last_Motifz = Last_Motifz[0]  # Weird Needed Step
        self.Last_Motifs = Last_Motifz


    def Locate_All_Last_Motifs(self):
        """Create Index for All Good Motifs
        Parameters:
        -----------
        .Song_Quality: list
            Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Locations: list
            Describes the Location of the Motif in the Bout, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Syl_Droplist
            Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
            [Number of Trials x 1 (numpy.unicode_)]
            ** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility**


        Returns:
        --------
        .Last_Motifs: list
            Index of All Last Motifs, this motif is the last motif in a bout they can be classified 'Good' or 'Bad'

        """
        # 3. Good Last Motifs
        # 3.1 Initialize Variables and Memory
        assert len(self.Song_Locations) == len(self.Song_Quality)
        All_Last_Holder = np.zeros(len(self.Song_Locations))  # Allocate Memory for Indexing

        # 3.2 Fill Logical for Good First Motifs
        for i in range(len(self.Song_Quality)):
            if self.Song_Locations[i][0] == 'Ending':  # Locate Desired Label Combination
                All_Last_Holder[i] = 1  # Mark them
        All_Last_Motifz = np.where(All_Last_Holder == 1)  # Create Index of Selected Label
        All_Last_Motifz = All_Last_Motifz[0]  # Weird Needed Step
        self.All_Last_Motifs = All_Last_Motifz

    def Locate_Good_Mid_Motifs(self):
        """Create Index for All Good Middle Motifs

        Parameters:
        -----------
        .Song_Quality: list
            Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Locations: list
            Describes the Location of the Motif in the Bout, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Syl_Droplist
            Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
            [Number of Trials x 1 (numpy.unicode_)]
            ** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility**


        Returns:
        --------
        .Good_Mid_Motifs: list
            Motifs that are classified as good and are 'Beginning' or 'End'
        """

        # 3. Good Last Motifs
        # 3.1 Initialize Variables and Memory
        assert len(self.Song_Locations) == len(self.Song_Quality)
        Good_Mid_Holder = np.zeros(len(self.Song_Locations))  # Allocate Memory for Indexing

        # 3.2 Fill Logical for Good First Motifs
        for i in range(len(self.Song_Quality)):
            if self.Song_Quality[i][0] == 'Good':
                if self.Song_Locations[i][0] != 'Ending' and self.Song_Locations[i][0] != 'Beginning':  # Locate Desired Label Combination
                    Good_Mid_Holder[i] = 1  # Mark them
        Good_Mid_Motifz = np.where(Good_Mid_Holder == 1)  # Create Index of Selected Label
        Good_Mid_Motifz = Good_Mid_Motifz[0]  # Weird Needed Step
        self.Good_Mid_Motifs = Good_Mid_Motifz

    def Locate_Bad_Full_Motifs(self):
        """Create Index for All Good Motifs
        Parameters:
        -----------
        .Song_Quality: list
            Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Locations: list
            Describes the Location of the Motif in the Bout, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Syl_Droplist
            Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
            [Number of Trials x 1 (numpy.unicode_)]
            ** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility**


        Returns:
        --------
        .Bad_Motifs: list
        Index of All Bad Motifs with no dropped syllables, These motifs have interferring audio noise
        """

        # 4. Bad Motifs w/ NO Dropped Syllables
        # 4.1 Initialize Variables and Memory
        assert len(self.Song_Locations) == len(self.Song_Syl_Drop)
        Bad_NDS_Holder = np.zeros(len(self.Song_Syl_Drop))  # Allocate Memory for Indexing

        # 4.2 Fill Logical for Bad Motifs (No Dropped Syllables)
        for i in range(len(self.Song_Quality)):
            if self.Song_Quality[i][0] == 'Bad':
                if self.Song_Syl_Drop[i][0] == 'None':  # Locate Desired Label Combination
                    Bad_NDS_Holder[i] = 1  # Mark them
        Bad_Motifz = np.where(Bad_NDS_Holder == 1)  # Create Index of Selected Label
        Bad_Motifz = Bad_Motifz[0]  # Weird Needed Step
        self.Bad_Motifs = Bad_Motifz

    def Locate_Last_Syll_Dropped(self):
        """ Create Index for All Motifs with the last Syllable Dropped

        Parameters:
        -----------
        .Song_Quality: list
            Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Locations: list
            Describes the Location of the Motif in the Bout, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Syl_Droplist
            Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
            [Number of Trials x 1 (numpy.unicode_)]
            ** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility**


        Returns:
        --------
        .LS_Drop: list
            Index of All Bad Motifs with the last syllable dropped, These motifs are classified as Bad
        """

        # 5. Bad Motifs w/ LAST Syllable Dropped
        # 5.1 Initialize Variables and Memory
        LS_Drop_Holder = np.zeros(len(self.Song_Syl_Drop))  # Allocate Memory for Indexing

        # 5.2 Fill Logical for Bad Motifs (Last Syllable Dropped)
        for i in range(len(self.Song_Quality)):
            if self.Song_Quality[i][0] == 'Bad':
                if self.Song_Syl_Drop[i][0] == 'Last Syllable':  # Locate Desired Label Combination
                    LS_Drop_Holder[i] = 1  # Mark them
        LS_Dropz = np.where(LS_Drop_Holder == 1)  # Create Index of Selected Label
        LS_Dropz = LS_Dropz[0]  # Weird Needed Step
        self.LS_Drop = LS_Dropz

    def Locate_Bouts(self):
        """ Create Index for All First Motifs

        Parameters:
        -----------
        .Song_Quality: list
            Describes the quality of the Motif. Options:['Good', 'Bad', 'NM': Not Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Locations: list
            Describes the Location of the Motif in the Bout, Options:['None', 'Beginning': First Motif, 'Ending': Last Motif]
            [Number of Trials x 1 (numpy.unicode_)]
        .Song_Syl_Droplist
            Describes Identity of which Syllable is dropped, Options:['None': Nothing Dropped, 'First Syllable', 'Last Syllable']
            [Number of Trials x 1 (numpy.unicode_)]
            ** This Annotation is mainly used for z020, may be deprecated in the future or update for more flexibility**

        Returns:
        --------
        .All_First_Motifs: list
            Index of All First Motifs in a Bout Regardless of Quality label, This is Useful for Clip-wise (Series) Analysis
        """

        # 6. All First Motifs
        # 6.1 Initialize Variables and Memory
        assert len(self.Song_Locations) == len(self.Song_Quality)
        All_First_Holder = np.zeros(len(self.Song_Locations))  # Allocate Memory for Indexing

        # 6.2 Fill Logical for All First Motifs
        for i in range(len(self.Song_Quality)):
            if self.Song_Locations[i][0] == 'Beginning':  # Locate Desired Label Combination
                All_First_Holder[i] = 1  # Mark them
        All_First_Motifz = np.where(All_First_Holder == 1)  # Create Index of Selected Label
        All_First_Motifz = All_First_Motifz[0]  # Weird Needed Step
        self.All_First_Motifs = All_First_Motifz

    #         self.Song_Quality = Labels_Quality
    #         self.Song_Locations = Labels_Location
    #         self.Song_Syl_Drop = Labels_Syl_Drop

    def Describe(self):
        """Describe relevant shorthand information about this particular trial

        Prints:
        -------
        Bird ID
        Trial Date
        Number of Examples of Song
        Number of Examples of Silence
        Number of Good Trials
        """
        print('Bird Id: ' + self.bird_id)
        print('Recording Date: ' + self.date)
        print('')
        print('Stats:')
        print('# of Motifs Total: ' + str(self.Num_Motifs))
        print('# of Silences Total: ' + str(self.Num_Silence))
        print('# of Good Motifs: ' + str(len(self.Good_Motifs)))
        print('# of Good First Motifs: ' + str(len(self.First_Motifs)))
        print('# of Bouts Total: ' + str(len(self.All_First_Motifs)))

    def Help(self):
        """Describe the Function and Revelant tools for using it
        """
        print('Hello Here is a walk through of this Function (Under Development)')
        print('The Initializing code does all of the Heavy Lifting If you would like more information use .Describe()')
