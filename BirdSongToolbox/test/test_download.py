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
