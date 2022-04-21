import typing

import pandas as pd

from mci_model import __version__ as _version
from mci_model.config.base import config
from mci_model.utilities.data_manager import load_pipeline
from mci_model.utilities.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_mci_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: typing.Union[pd.DataFrame, dict],) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _mci_pipe.predict(X=validated_data)
        results = {
            "predictions": [list(pred) for pred in predictions],  # type: ignore
            "version": _version,
            "errors": errors,
        }

    return results
