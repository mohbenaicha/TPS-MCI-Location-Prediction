
import pytest


from mci_model.config.base import config, DATASET_DIR
from mci_model.utilities.data_manager import load_dataset


@pytest.fixture(scope="function")
def test_input_data():
    return load_dataset(file_name=f'{DATASET_DIR}/'+config.app_config.test_data_file,
        training=False,
        )

@pytest.fixture(scope="session")
def config_dir():
    pass

@pytest.fixture(scope="session")
def test_pipeline_data():
    file_path = DATASET_DIR / str(config.app_config.training_data_file)
    return load_dataset(file_name=file_path)