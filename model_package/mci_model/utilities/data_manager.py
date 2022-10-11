import logging
import typing as t
from pathlib import Path
import joblib
import json
import pandas as pd
from sklearn.pipeline import Pipeline

from mci_model import __version__ as _version
from mci_model.config.base import DATASET_DIR, TRAINED_MODEL_DIR, config

_logger = logging.getLogger(__name__)
m_config = config.model_config

def load_dataset(*, file_name: str, 
    training: bool = True,
    drop_duplicates=True,
    drop_na_and_zeroes=True,
    drop_nsa=True) -> pd.DataFrame:

    data = pd.read_csv(file_name)
    
    # This is a workaround to teach the training pipeline to accept new features
    # during inference as a given, where as in training, they are engineered
    # In production, inference features are inputted into the model
    # at the discretion of those utilizing the crime location prediction application.

    if training:
        inference_features_to_add = m_config.inference_features_to_add
        for feat in inference_features_to_add:
            data[feat] = 0
    try:
        data = drop_invalid_data(
                                  data=data,
                                  drop_duplicates=drop_duplicates,
                                  drop_na_and_zeroes=drop_na_and_zeroes,
                                  drop_nsa=drop_nsa)
    except Exception as error:
        _logger.error(error)
        
    return data[m_config.train_features + m_config.inference_features_to_add + m_config.targets]


def drop_invalid_data(data,
                      drop_duplicates=True,
                      drop_na_and_zeroes=True,
                      drop_nsa=True):
    ''' drops records that are deemed not fit for training as justified in the research notebook''' 
    
    if drop_duplicates and m_config.duplicate_record_key in data.columns:
        data.drop_duplicates(
            subset=[str(m_config.duplicate_record_key)], 
            inplace=True
        )

    # MCI data will have recent reports with Lat/Long
    # with coordinate of 0, 0 respectively whereas normally
    # so they should be dropped for training purposes

    if drop_na_and_zeroes:

        data.dropna(subset= [feature for feature in m_config.features_na_not_allowed if feature in data.columns], inplace=True)
        
        trgt_idx = (list(data[data[m_config.targets[0]] == 0].index) + 
                     list(data[data[m_config.targets[1]] == 0].index))
        data.drop(axis=0, index = trgt_idx, inplace=True)

    if drop_nsa:
        for feat in m_config.NSA_features:
            if feat in data.columns:
                NSA_idx = data[data[feat] == 'NSA'].index
                data.drop(index=NSA_idx, inplace=True)

    return data


def load_data_to_engineer_features(path: str = DATASET_DIR, 
        config: t.List[str] = config) -> (t.List[str], t.Dict[str, t.List[t.Union[int, float]]]):
    
    added_feature_names = [config.model_config.engineered_features['pub'], 
                            config.model_config.engineered_features['park'], 
                            config.model_config.engineered_features['police_station']]
    added_features = [json.load(open(Path(f'{path}/{name}.json'))) for name in added_feature_names]

    return added_feature_names, added_features


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
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
    
    # As explained in the research notebook, in production, the model
    # takes certain inputs that are not learned during trained but fed into the 
    # pipeline for inference based on the discretion of the users of the model
    # hence, the feature addition step in the pipeline is remove since it's only
    # used in training

    trained_model.steps.pop(5)

    return trained_model
