import typing
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from mci_model import __version__ as _version
from mci_model.config.base import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    data = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    try:
        data = data.drop_duplicates(
            subset=[str(config.model_config.column_to_drop)]
        ).reset_index()
    except Exception:
        print("Column to drop does not exist.")

    # the decision to drop null values this isn't arbitrary since we've decided
    # to drop these since discovering that null values are not records of crimes
    # reported long after they were committed and so don't have accurate
    # occurrence date-time information
    data.dropna(axis=0, inplace=True)

    return data


def remove_old_pipelines(*, files_to_keep: typing.List[str]) -> None:
    """
    Iterates through every file in the target directory and removes all but the
    new pipeline file and the __init__.py file.

    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    # define name pipeline of newely trained model
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)  # save new pipeline


def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model
