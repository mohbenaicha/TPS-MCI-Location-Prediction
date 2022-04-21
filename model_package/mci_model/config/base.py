from pathlib import Path
from typing import Dict, List

# import mci_model
from pydantic import BaseModel
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
    targets: List[str]
    features: List[str]
    levels: Dict[str, List[int]]
    holidays: List[int]
    seasons: Dict[str, List[int]]
    column_to_drop: str
    test_size: float
    random_state: int


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
