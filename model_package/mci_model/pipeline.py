from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from typing import List
from pydantic import BaseModel

from mci_model.config.base import config, DATASET_DIR
from mci_model.utilities import transformers as t
from mci_model.utilities.data_manager import load_data_to_engineer_features as load_data


m_config = config.model_config

added_feature_names, added_features = load_data(
    config = config,
    path = DATASET_DIR
    )

hyperparams = m_config.hyperparams

mci_pipeline = Pipeline(
    [
        ("impute_date_time", t.DateTimeImputer(m_config.datetime_features['date'])),
        ("add_weekday/weekend", t.WeekdayTransformer()),
        ("add_ToD_crime_level", t.ToDTransformer(levels=m_config.levels)),
        ("add_seasons", t.SeasonTransformer(seasons=m_config.seasons)),
        ("add_holidays", t.HolidayTransformer(holidays=m_config.holidays)),
        ("add_new_features", t.NewFeatureTransformer(added_features, added_feature_names)),
        ("OHE", OneHotEncoder(sparse=False)),
        ("PCA", PCA(n_components=hyperparams['pca']['n_components'], svd_solver= str(hyperparams['pca']['svd_solver']))),
        ("linear_regression", LinearRegression()),
    ]
)
