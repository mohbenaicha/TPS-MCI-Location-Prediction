import pytest

from mci_model.config.base import config
from mci_model.utilities.data_manager import load_dataset


@pytest.fixture()
def test_input_data():
    return load_dataset(file_name=config.app_config.test_data_file)
