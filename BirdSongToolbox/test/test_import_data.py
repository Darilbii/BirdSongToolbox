""" Test the import_data module"""

import pytest
from pathlib import Path
from BirdSongToolbox.config.utils import CONFIG_PATH

from BirdSongToolbox.import_data import ImportData


@pytest.mark.run(order=1)
@pytest.fixture()
def bird_id():
    return 'z007'

@pytest.mark.run(order=1)
@pytest.fixture()
def session():
    return 'day-2016-09-09'

@pytest.mark.run(order=1)
def test_import_data_default(bird_id, session, create_dummy_config):
    Data = ImportData(bird_id=bird_id, session=session)

@pytest.mark.run(order=1)
def test_import_data_specified(bird_id, session, chunk_data_path):
    Data = ImportData(bird_id=bird_id, session=session, location=chunk_data_path)
