""" Test Script that Downloads Test Data for the Test Scripts"""
import pytest
# from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from BirdSongToolbox.config.settings import DATA_DIR, TEST_DATA_ZIP, TEST_DATA_DIR

# PROJECT_DIR = Path(__file__).resolve().parents[1]
# DATA_DIR = PROJECT_DIR / "data"
# TEST_DATA_ZIP = DATA_DIR / "PrePd_Data.zip"
# TEST_DATA_DIR = DATA_DIR / "PrePd_Data"


TEST_DATA_PIPELINE_ZIP = DATA_DIR / "Pipeline_Test_Dataz020_day-2016-06-02.pckl.zip"
TEST_DATA_PIPELINE_PCKL = DATA_DIR / "Pipeline_Test_Dataz020_day-2016-06-02.pckl"

TEST_DATA_PRAAT_UTILS_ZIP = DATA_DIR / "praat_utils_test_data.zip"
TEST_DATA_PRAAT_UTILS_DIR = DATA_DIR / "Chunk_TextGrids_Final"

# For Travis There needs the config file must be created for further tests
# local_data_path = input("What is the path to the data folder on your local computer?)")
#
# # Verify that this local path exists
# verify = Path(local_data_path)
# verify.resolve()
#
# if verify.exists():
#     # Create the setting.pckl file
#     default_path.resolve()
#     with default_path.open(mode='wb') as settings_file:
#         pk.dump(local_data_path, settings_file,
#                 protocol=0)  # Protocol 0 is human readable and backwards compatible

@pytest.mark.run(order=0)
def test_download_data():
    if not TEST_DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        gdd.download_file_from_google_drive(file_id='18rzzBSSIIIHOL8Ot27bCqipTdYZlxBbO',
                                            dest_path=TEST_DATA_ZIP.as_posix(),
                                            unzip=True)
    # Test for Test Data
    assert TEST_DATA_DIR.exists()

# TODO: Fix confusion between parameter name Features (Epoch_Analysis_Tools) and Channels (PreProcessClass) [They are the same and refer to Neural objects of the Pipeline() Class]
# TODO: Update the Pipeline() test data to follow a standard pre-process framework to standardize tests
# TODO: Write test scripts for Epoch_Analysis_Tools using a updated Version of the Pipeline test data

@pytest.mark.run(order=0)
def test_download_data_instance_of_pipeline_class():
    """
    Code Used To Generate Data:

    # # Import Data
    # >>> z020_day_1 = tb.Import_PrePd_Data('z020', 'day-2016-06-02')
    # # Pre-Process Data
    # >>> Pipe_1= tb.Pipeline(z020_day_1)
    #
    # # Run Pre-Process Steps
    # >>>Pipe_1.Define_Frequencies(([8],[13]))
    # >>>Pipe_1.Band_Pass_Filter(verbose = False)
    # # Pipe_1.Re_Reference()
    # >>>Pipe_1.Z_Score()
    #
    # >>>Pipe_1.Pipe_end()
    #
    # >>>bird_id = 'z020'
    # >>>sess_name = 'day-2016-06-02'
    #
    # >>>file_name = 'Pipeline_Test_Data' + bird_id + '_' + sess_name + '.pckl'
    # >>>Destination = '/home/debrown/' + file_name
    #
    # # f = open(Destination, 'wb')
    # # pickle.dump(Pipe_1, f)
    # # f.close()
    #
    # # f = open(Destination, 'rb')
    # # obj = pickle.load(f)
    # # f.close()

    """

    if not TEST_DATA_PIPELINE_PCKL.is_file():
        gdd.download_file_from_google_drive(file_id='1d6KScMOawxATEQL8sqQzcKDvRgKSCOJQ',
                                            dest_path=TEST_DATA_PIPELINE_ZIP.as_posix(),
                                            unzip=True)
        # Test for Pipeline() Test Data
        assert TEST_DATA_DIR.exists()

@pytest.mark.run(order=0)
def test_download_data_chunk_textgrid_directory():
    """Download Data for testing the praat_utils module"""
    if not TEST_DATA_PRAAT_UTILS_DIR.exists():
        gdd.download_file_from_google_drive(file_id='1G8cGCJzczptIon9kus0m1X417HUA5QUE',
                                            dest_path=TEST_DATA_PRAAT_UTILS_ZIP.as_posix(),
                                            unzip=True)
        # Test for annotate module Test Data
        assert TEST_DATA_PRAAT_UTILS_DIR.exists()

