from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

from mci_model.config.base import config

m_config = config.model_config

def validate_inputs(
    *, input_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Validate inputs are as expected according to a defined
    Pydantic schema."""

    validated_data = input_data[m_config.train_features + 
                                m_config.inference_features_to_add].copy()
    errors = None

    try:
        MultipleMCIDataInputs(inputs=validated_data.to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


# for testing prediction
class MCIDataInputSchema(BaseModel):
    occurrencedate: Optional[str] # used to engineer timedate features
    premises_type: Optional[str]
    occurrencehour: Optional[int]
    occurrenceday: Optional[float]
    occurrencedayofyear: Optional[float]
    occurrencemonth: Optional[str]
    occurrencedayofweek: Optional[str]
    MCI: Optional[str]
    Neighbourhood: Optional[str]
    Pub_Id: Optional[int]
    Park_Id: Optional[int]
    PS_Id: Optional[int]


class MultipleMCIDataInputs(BaseModel):
    inputs: List[MCIDataInputSchema]
