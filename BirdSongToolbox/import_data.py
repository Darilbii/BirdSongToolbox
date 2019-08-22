""" This Module contains a Class used for importing Chunk'd Behavioral and Neural Data and its relevant meta data

"""

from BirdSongToolbox.config.settings import CHUNKED_DATA_PATH
from BirdSongToolbox.file_utility_functions import _load_pckl_data,_load_numpy_data, _load_json_data

from pathlib import Path
import warnings


class ImportData:
    """ Imports Data Specified by user into a Object Oriented Structure

    Parameters
    ----------
    bird_id : str
        Bird identifier to locate specified bird's data folder
    session : str
        Experiment day to locate it's Folder
    data_type : string
        String directing the type of neural signal to import, (Options: 'LPF_DS', 'LPF', 'Raw')
    location : str or Path, optional
        Location to search for the data other than default CHUNKED_DATA_PATH (Optional if default path configured)


    Attributes
    ----------
    bird_id : str
        Bird identifier to locate specified bird's data folder
    date : str
        Experiment day to locate it's Folder
    song_neural : list | shape = [Chunk]->(channels, Samples)
        Neural data that is Low-Pass Filter at 400 Hz and Downsampled to 1 KHz, list of 2darrays
    song_audio : list | shape = [Chunk]->(channels, Samples)
        Audio data that is Bandpass Filtered between 300 and 10000 Hz, list of 2darrays
    song_index : list | shape = [Chunk]->(absolute start, absolute end)
        List of the Absolute Start and End of Each Chunk for that Recordings Day
    song_ledger : list | shape = [Chunk]->(first epoch, ..., last epoch)
        Ledger of which epochs occur in each Chunk, Chunks that only contain one Epoch have a length of 1
    kwe_times : array
        array of the absolute start of the labels
    kwe_epoch_times : array, | shape = (num_motifs, 2)
        Absolute start and end time of the old epochs
        (Start Sample, End Sample)

    """

    def __init__(self, bird_id: str, session: str, data_type='LPF_DS', location=None):

        # Initialize Session Identifiers
        self.bird_id = bird_id
        self.date = session
        self.data_type = data_type

        # Resolve Location of directory to be imported
        self._data_folder = self._resolve_source(location=location)   # Directory of Data to Import

        # Import Song Data
        self.song_neural = self._get_data_from_pckl(data_name="Large_Epochs_Neural", behave_type="Song")  # Neural Data
        self.song_audio = self._get_data_from_pckl(data_name="Large_Epochs_Audio", behave_type="Song")  # Audio Data
        self.song_index = self._get_data_from_pckl(data_name="Large_Epochs_Times", behave_type="Song")  # Absolute Times
        self.song_ledger = self._get_data_from_pckl(data_name="Epochs_Ledger", behave_type="Song")  # Ledger of Events

        # Import Song Handlabels
        self.song_handlabels = self._get_data_from_pckl(data_name="chunk_handlabels", behave_type="Song")  # handlabels

        # Import Old Song Epoch MetaData
        self.kwe_times = self._get_data_from_npy(data_name="AbsoluteTimes", behave_type="Song")  # KWE Times
        self.kwe_epoch_times = self._get_data_from_npy(data_name="EpochTimes", behave_type="Song")  # KWE Epoch Times

    def _resolve_source(self, location=None):
        """Resolve Source of Data"""

        # Basic Setup for path Creation
        if location is None:
            if CHUNKED_DATA_PATH == '':
                warnings.warn('There is currently No default Paths for BirdSongToolbox. Please update the configuration'
                              ' if you want to use the default paths convenience api of BirdSongToolbox')
            experiment_folder = CHUNKED_DATA_PATH
        else:
            experiment_folder = location

        if isinstance(experiment_folder, str):
            experiment_folder = Path(experiment_folder)
            assert experiment_folder.exists(), "Directory pointed by location parameter does not exist"
            experiment_folder.resolve()

        return experiment_folder

    def _get_data_from_pckl(self, data_name: str, behave_type: str):
        """Convenience Function to Pull Relevant Data"""

        assert behave_type in ["Song", "Silence"]

        full_data_name = data_name + '_' + behave_type
        return _load_pckl_data(data_name=full_data_name, bird_id=self.bird_id, session=self.date,
                               source=self._data_folder)
    def _get_data_from_npy(self, data_name: str, behave_type: str):
        """Convenience Function to Pull Relevant Data"""

        assert behave_type in ["Song", "Silence"]

        full_data_name = data_name + '_' + behave_type
        return _load_numpy_data(data_name=full_data_name, bird_id=self.bird_id, session=self.date,
                                source=self._data_folder)





