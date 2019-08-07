
from shutil import rmtree
import pytest
from pathlib import Path
from BirdSongToolbox.config.utils import CONFIG_PATH
from BirdSongToolbox.config.utils import get_spec_config_path, update_config_path

@pytest.fixture(scope="module")
def create_dummy_config():
    from BirdSongToolbox.config.utils import _make_default_config_file

    _make_default_config_file()

    yield

    CONFIG_PATH.unlink()



@pytest.mark.run(order=0)
@pytest.mark.parametrize("specific_path", ["Chunked_Data_Path", pytest.param("wrong_path", marks=pytest.mark.xfail)])
def test_get_spec_config_path(specific_path):
    get_spec_config_path(specific_path=specific_path)


@pytest.mark.run(order=0)
@pytest.mark.parametrize("specific_path,new_path",
                         [("Chunked_Data_Path", CONFIG_PATH),
                          ("Chunked_Data_Path", str(CONFIG_PATH)),
                          pytest.param("Chunked_Data_Path", "data/folder", marks=pytest.mark.xfail),
                          pytest.param("wrong_path", "data/folder", marks=pytest.mark.xfail)])
def test_update_config_path(specific_path, new_path, create_dummy_config):
    update_config_path(specific_path=specific_path, new_path=new_path)

    result = get_spec_config_path(specific_path=specific_path)

    if isinstance(new_path, Path):
        new_path.resolve()
    assert result == str(new_path)
