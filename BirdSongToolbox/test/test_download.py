""" Test Script that Downloads Test Data for the Test Scripts"""
import pytest
from google_drive_downloader import GoogleDriveDownloader as GdD

# TODO: Fix confusion between parameter name Features (Epoch_Analysis_Tools) and Channels (PreProcessClass) [They are the same and refer to Neural objects of the Pipeline() Class]
# TODO: Update the Pipeline() test data to follow a standard pre-process framework to standardize tests
# TODO: Write test scripts for Epoch_Analysis_Tools using a updated Version of the Pipeline test data


@pytest.mark.run(order=0)
def test_download_data(data_path, PrePd_data_dir_path):
    PrePd_data_zip_path = data_path / "PrePd_Data.zip"

    if not PrePd_data_dir_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        GdD.download_file_from_google_drive(file_id='18rzzBSSIIIHOL8Ot27bCqipTdYZlxBbO',
                                            dest_path=PrePd_data_zip_path.as_posix(),
                                            unzip=True)
    # Test for Test Data
    assert PrePd_data_dir_path.exists()

@pytest.mark.run(order=0)
def test_download_data_instance_of_pipeline_class(data_path, data_pipeline_class_pckl_path):
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

    data_pipeline_class_zip_path = data_path / "Pipeline_Test_Dataz020_day-2016-06-02.pckl.zip"

    if not data_pipeline_class_pckl_path.is_file():
        GdD.download_file_from_google_drive(file_id='1d6KScMOawxATEQL8sqQzcKDvRgKSCOJQ',
                                            dest_path=data_pipeline_class_zip_path.as_posix(),
                                            unzip=True)
    # Test for Pipeline() Test Data
    assert data_pipeline_class_pckl_path.exists()


@pytest.mark.run(order=0)
def test_download_data_chunk_textgrid_directory(data_path, data_praat_utils_dir_path):
    """Download Data for testing the praat_utils module"""

    data_praat_utils_zip_path = data_path / "praat_utils_test_data.zip"

    if not data_praat_utils_dir_path.exists():
        GdD.download_file_from_google_drive(file_id='1G8cGCJzczptIon9kus0m1X417HUA5QUE',
                                            dest_path=data_praat_utils_zip_path.as_posix(),
                                            unzip=True)
    # Test for annotate module Test Data
    assert data_praat_utils_dir_path.exists()

@pytest.mark.run(order=0)
def test_download_data_chunk_data_demo_directory(data_path, chunk_data_path):
    """Download Data for testing the praat_utils module"""

    test_data_chunk_data_demo_zip = data_path / "Chunk_Data_Demo.zip"
    # test_data_chunk_data_demo_dir = data_path / "Chunk_Data_Demo"
    test_data_chunk_data_demo_dir = chunk_data_path

    if not chunk_data_path.exists():
        GdD.download_file_from_google_drive(file_id='1GLREIG8zaW3-4MJLVg9OibfLBBKV28pp',
                                            dest_path=test_data_chunk_data_demo_zip.as_posix(),
                                            unzip=True)
    # Test for annotate module Test Data
    assert chunk_data_path.exists()
