from typing import Any, List, Optional

from mci_model.utilities.validation import MCIDataInputSchema
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[List[float]]]


class MultipleMCIDataInputs(BaseModel):
    inputs: List[MCIDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "occurrencehour": 11,
                        "Pub_Id": 136,
                        "Park_Id": 43,
                        "PS_Id": 22,
                        "premises_type": "House",
                        "occurrencemonth": "July",
                        "occurrencedayofweek": "Sunday",
                        "MCI": "Assault",
                        "Neighbourhood": "Eglinton East (138)",
                        "occurrenceday": 14.0,
                        "occurrencedayofyear": 195.0,
                    }
                ]
            }
        }
