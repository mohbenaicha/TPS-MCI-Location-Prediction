from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

from mci_model.config.base import config


def validate_inputs(
    *, input_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Validate inputs are as expected according to a defined
    Pydantic schema."""

    validated_data = input_data[config.model_config.features].copy()
    errors = None

    try:
        MultipleMCIDataInputs(inputs=validated_data.to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


# for testing prediction
class MCIDataInputSchema(BaseModel):

    occurrencehour: Optional[int]
    Pub_Id: Optional[int]
    Park_Id: Optional[int]
    PS_Id: Optional[int]
    premises_type: Optional[str]
    occurrencemonth: Optional[str]
    occurrencedayofweek: Optional[str]
    MCI: Optional[str]
    Neighbourhood: Optional[str]
    occurrenceday: Optional[float]
    occurrencedayofyear: Optional[float]


class MultipleMCIDataInputs(BaseModel):
    inputs: List[MCIDataInputSchema]
