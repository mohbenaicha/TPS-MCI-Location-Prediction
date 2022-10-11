import json
import multiprocessing as mp
from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from mci_model.config.base import config

    
_dt_features = config.model_config.datetime_features # add model config key
_eng_features = config.model_config.engineered_features # # add model config key


class DateTimeImputer(BaseEstimator, TransformerMixin):
    ''' Extracts day of the year, day of theweek and ordinal value of month
    from the occurencedate feature'''
    def __init__(self, date_col: str):
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        X[_dt_features['day_of_year']] = X[self.date_col]. \
            astype('datetime64').apply(lambda x: x.day_of_year)
        X[_dt_features['month']] = X[self.date_col]. \
            astype('datetime64').apply(lambda x: x.month)
        X[_dt_features['day_of_week']] = X[self.date_col]. \
            astype('datetime64').apply(lambda x: x.day_of_week)
        X[_dt_features['day_of_month']] = X[self.date_col]. \
            astype('datetime64').apply(lambda x: x.day)
       
        # occurence date is no longer required, drop it and return X
        X.drop(columns=[_dt_features['date']], inplace=True, axis=1)
        
        return X


class SeasonTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        seasons: dict = {"winter": None, "spring": None, "summer": None, "fall": None},
    ):
        self.seasons = seasons

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        feature_name = _eng_features['season']
        
        X.loc[
            (X[_dt_features['day_of_year']] >= int(self.seasons["winter"][0]))
            | (X[_dt_features['day_of_year']] <= int(self.seasons["winter"][1])),
            feature_name,
        ] = "Winter"

        X.loc[
            (X[_dt_features['day_of_year']] >= int(self.seasons["spring"][0]))
            & (X[_dt_features['day_of_year']] <= int(self.seasons["spring"][1])),
            feature_name,
        ] = "Spring"

        X.loc[
            (X[_dt_features['day_of_year']] >= int(self.seasons["summer"][0]))
            & (X[_dt_features['day_of_year']] <= int(self.seasons["summer"][1])),
            feature_name,
        ] = "Summer"

        X.loc[
            (X[_dt_features['day_of_year']] >= int(self.seasons["fall"][0]))
            & (X[_dt_features['day_of_year']] <= int(self.seasons["fall"][1])),
            feature_name,
        ] = "Fall"
        return X


class WeekdayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.loc[
            (X[_dt_features['day_of_week']] == 0)
            | (X[_dt_features['day_of_week']] == 6),
            _eng_features['weekday'],
        ] = "weekend"

        X.loc[
            (X[_dt_features['day_of_week']] != 0)
            & (X[_dt_features['day_of_week']] != 6),
            _eng_features['weekday'],
        ] = "weekday"
        return X


class ToDTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, levels: dict):
        if not isinstance(levels, dict) or not all(
            [isinstance(level, list) for level in levels.values()]
        ):
            raise ValueError(
                "levels should be a dictionary of key[string]: value[list] pairs"
            )
        self.levels = levels

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for key, values in zip(self.levels.keys(), self.levels.values()):
            X.loc[X[_dt_features['hour']].isin(values), _eng_features['ToD']] = key
        return X


class HolidayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, holidays):
        if not isinstance(holidays, list) or not all(
            [isinstance(element, int) for element in holidays]
        ):
            raise ValueError("holidays should be a list of integers")
        self.holidays = holidays

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Holiday"] = X[_dt_features['day_of_year']].apply(
            lambda i: "holiday" if i in self.holidays else "non-holiday"
        )
        return X


# This is not used since filling in NSA values causes serious skews in 
# predictions and because NSA records to not apply to TPS jurisdiction
# but it's left here to demonstrate how missing features can be handled


class ReplaceNSA(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nsa_column = "Neighbourhood"
        self.targets = ["Long", "Lat"]
        self.distance = 1000

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.XY = pd.concat((X, y), axis=1)
        self.cluster_centres = pd.concat(
            (
                self.XY.groupby(self.nsa_column)[y.columns[0]].mean(),
                self.XY.groupby(self.nsa_column)[y.columns[1]].mean(),
            ),
            axis=1,
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.nsa_records = self.XY[self.XY[self.nsa_column] == "NSA"][self.targets]

        for row, index in zip(self.nsa_records.values, self.nsa_records.index):
            for i, cc in enumerate(self.cluster_centres.values):
                new_distance = (row[0] - cc[0]) ** 2 + (row[1] - cc[1]) ** 2
                if new_distance < self.distance:
                    self.distance = new_distance
                    shortest_idx = i
            X[self.nsa_column][index] = self.cluster_centres.index[shortest_idx]
            self.distance = 1000
        return X


class NewFeatureTransformer(BaseEstimator, TransformerMixin):


    def __init__(self, features: List[object], feature_names: List[str]):
        self.features = features
        self.feature_names = feature_names


    def _process_partition(self, *args):
        new_col = []
        for x in range(len(args[0][0])):
            distances = []
            for y in range(len(args[0][1])):
                distance = (
                    (args[0][1][y, 0] - args[0][0][x, 0]) ** 2
                    + (args[0][1][y, 1] - args[0][0][x, 1]) ** 2
                ) ** 0.5
                distances.append(distance)
            new_col.append(distances.index(min(distances)))
        return new_col

    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.target = y.copy().reindex(columns=["Lat", "Long"]).values
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        target = self.target
        X_copy = X.copy()
        cores = mp.cpu_count()
        # mp.set_start_method('spawn')
        # Brute force algorithm used for training purposes.
        # New Feature Transformer is not used in inference due to reasons
        # explained in the research notebook
        for feature, name in zip(self.features, self.feature_names):
            X_copy[name] = 0
            source = pd.DataFrame(feature).set_index(name).values
            data_split = np.array_split(self.target, cores)

            args = [(part, source) for part in data_split]

            with mp.Pool(cores) as pool:
                 results = []
                 result = pool.map(self._process_partition, args)
                 extend = results.extend
                 [extend(_list) for _list in result]
                 X_copy[name] = results


        return X_copy
