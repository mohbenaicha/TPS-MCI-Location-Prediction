import math

import pandas as pd
from fastapi.testclient import TestClient
from mci_model.config.base import config


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given
    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": test_data.drop_duplicates(
            subset=[str(config.model_config.column_to_drop)]
        )
        .reset_index()
        .to_dict(orient="records")
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    result = response.json()

    expected_first_prediction_value = [-79.43863713, 43.6423412]
    expected_no_predictions = 46944

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
