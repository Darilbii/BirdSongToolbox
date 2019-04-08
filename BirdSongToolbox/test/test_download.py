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


@pytest.mark.run(order=0)
def test_download_Pipeline_date():

    if not TEST_DATA_PIPELINE_PCKL.is_file():
        gdd.download_file_from_google_drive(file_id='1d6KScMOawxATEQL8sqQzcKDvRgKSCOJQ',
                                            dest_path=TEST_DATA_PIPELINE_ZIP.as_posix(),
                                            unzip=True)
        # Test for Pipeline() Test Data
        assert TEST_DATA_DIR.exists()
