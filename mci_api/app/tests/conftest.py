from typing import Generator
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mci_model.config.base import config, DATASET_DIR
from mci_model.utilities.data_manager import load_dataset

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    file_path = DATASET_DIR / f'{config.app_config.test_data_file}'
    return load_dataset(
        file_name = file_path,
        training = False)


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
