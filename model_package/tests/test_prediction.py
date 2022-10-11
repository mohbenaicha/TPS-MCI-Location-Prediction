import math
import numpy
from mci_model.predict import make_prediction


def test_make_prediction(test_input_data):
    # Given
    expected_first_prediction_value = numpy.array([ -79.181186, 43.7905 ])
    expected_no_predictions = 30000

    # When
    result = make_prediction(input_data=test_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0][0], numpy.float64) and isinstance(
        predictions[0][1], numpy.float64
    )
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(
        predictions[0][1], expected_first_prediction_value[0], abs_tol=0.05
    ) and math.isclose(
        predictions[0][0], expected_first_prediction_value[1], abs_tol=0.05
    )
