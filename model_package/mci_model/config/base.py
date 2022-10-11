from pathlib import Path
from typing import Dict, List, Union, Optional

# import mci_model
from pydantic import BaseModel, validator
from strictyaml import YAML, load

PACKAGE_ROOT = Path(__file__).parent.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "fitted_models"


# configurations are divided into application and model configurations,
# then combined into a single config class that enherits from the
# BaseModel class


class AppConfig(BaseModel):
    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    duplicate_record_key: str
    datetime_features: Dict[str, str]
    features_na_not_allowed: List[str]
    NSA_features: List[str]
    engineered_features: Dict[str, str]
    train_features: List[str]
    features_to_drop: List[str]
    targets: List[str]
    inference_features_to_add: List[str]
    model_controls: Dict[str, Union[float, List[str]]]
    hyperparams: Dict[str, Dict[str, Union[int, float, str]]]
    test_size: float
    random_state: int
    levels: Dict[str, List[int]]
    holidays: List[int]
    seasons: Dict[str, List[int]]

    @validator('hyperparams')
    def validate_model_controls(cls, value, values):
        allowed_estimators_ = values['model_controls']['allowed_estimators']
        if value['estimator']['name'] in allowed_estimators_:
            return value
        raise ValueError(f"{value['estimator']['name']} not allowed.")

        min_var = values['model_controls']['min_pca_var']
        if value['pca']['n_components'] >= min_var:
            return value
        raise ValueError(f'n_components too low, specify a value above  {min_var}.')


class Config(BaseModel):
    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(
        f"""Config file not found at path:
                                    {CONFIG_FILE_PATH!r}"""
    )


def get_config(cfg_path: Path = None) -> YAML:
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as f:
            parsed_config = load(f.read())
            return parsed_config
    raise OSError(f"Config file not found at path: {cfg_path}")


def create_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        parsed_config = get_config()

# specify the data attribute from the strictyaml YAML type.
    config_ = Config(
        app_config=AppConfig(
            **parsed_config.data
        ),  # **parsed_config.data inpacks list of dictionaries
        model_config=ModelConfig(**parsed_config.data),
    )

    return config_


# python object containig configuration to be used in other scripts
config = create_config()
