import math
import json
import pytest

import pandas as pd
from fastapi.testclient import TestClient
from mci_model.config.base import config
from mci_model import __version__


@pytest.mark.integration
def test_health(client: TestClient) -> None:
    res = client.get('http://localhost:8001/api/v1/health') # get response to health endpoint
    assert res.status_code == 200
    assert res.json()['model_version'] == __version__


@pytest.mark.integration
def test_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given
    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": test_data.to_dict(orient="records")
    }

    # When
    res = client.post(     # post request to predict endpoint
        "http://localhost:8001/api/v1/predict",
        json=payload, # body
    )

    # Then
    assert res.status_code == 200
    result = res.json()

    expected_first_prediction_value = [ -79.181186, 43.7905 ]
    expected_no_predictions = 30000

    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0][0], float) and isinstance(predictions[0][1], float)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(
        predictions[0][1], expected_first_prediction_value[0], abs_tol=0.05
    ) and math.isclose(
        predictions[0][0], expected_first_prediction_value[1], abs_tol=0.05
    )
