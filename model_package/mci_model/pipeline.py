from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mci_model.config.base import config
from mci_model.utilities import transformers as t

mci_pipeline = Pipeline(
    [
        ("add_weekday/weekend", t.WeekdayTransformer()),
        ("add_ToD_crime_level", t.ToDTransformer(levels=config.model_config.levels)),
        ("add_seasons", t.SeasonTransformer(seasons=config.model_config.seasons)),
        ("add_holidays", t.HolidayTransformer(holidays=config.model_config.holidays)),
        ("OHE", OneHotEncoder(sparse=False)),
        ("PCA", PCA(n_components=0.95, svd_solver="full")),
        ("linear_regression", LinearRegression()),
    ]
)
