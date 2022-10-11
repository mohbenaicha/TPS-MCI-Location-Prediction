import math
import json
import pytest
import time
import requests
from requests_futures.sessions import FuturesSession
import pandas as pd
from mci_model.config.base import config, DATASET_DIR
from mci_model.utilities.data_manager import load_dataset
import numpy as np

def load_data():
    file_path = DATASET_DIR / f'{config.app_config.training_data_file}'
    return load_dataset(
        file_name = file_path,
        training = False)

def test_request(test_data: pd.DataFrame) -> None:
    session = FuturesSession(max_workers=1)
    res = []
    for i in range(10):
        payload = {
    	"inputs": test_data.loc[i:i+1000, :].replace({np.nan: None}).to_dict(orient="records")
        }
        # print(payload['inputs'][:5])
        session.post(
    	url="http://localhost:8001/api/v1/predict",
    	json= payload
        )
        print(f"request {i} sent*********")
    return res


if __name__ == '__main__':
   data = load_data()
   test_request(data)
