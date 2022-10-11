import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
        X.loc[
            (X["occurrencedayofyear"] >= int(self.seasons["winter"][0]))
            | (X["occurrencedayofyear"] <= int(self.seasons["winter"][1])),
            "Season",
        ] = "Winter"

        X.loc[
            (X["occurrencedayofyear"] >= int(self.seasons["spring"][0]))
            & (X["occurrencedayofyear"] <= int(self.seasons["spring"][1])),
            "Season",
        ] = "Spring"

        X.loc[
            (X["occurrencedayofyear"] >= int(self.seasons["summer"][0]))
            & (X["occurrencedayofyear"] <= int(self.seasons["summer"][1])),
            "Season",
        ] = "Summer"

        X.loc[
            (X["occurrencedayofyear"] >= int(self.seasons["fall"][0]))
            & (X["occurrencedayofyear"] <= int(self.seasons["fall"][1])),
            "Season",
        ] = "Fall"
        return X


class WeekdayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.loc[
            (X["occurrencedayofweek"] == "Sunday")
            | (X["occurrencedayofweek"] == "Saturday"),
            "Weekend",
        ] = "weekend"

        X.loc[
            (X["occurrencedayofweek"] != "Sunday")
            & (X["occurrencedayofweek"] != "Saturday"),
            "Weekend",
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
            X.loc[X["occurrencehour"].isin(values), "ToDCrimeLevel"] = key
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
        X["Holiday"] = X["occurrencedayofyear"].apply(
            lambda i: "holiday" if i in self.holidays else "non-holiday"
        )
        return X



class ReplaceNSA(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nsa_column = "Neighbourhood"
        self.targets = ["Long", "Lat"]
        self.distance = 1000

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
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
    def __init__(self, features: list[object], feature_names: list[str]):
        self.features = features
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        # we use numpy arrays instead of dataframes given the intesity of the
        # transformations
        target = y.copy().reindex(columns=["Lat", "Long"]).values
        X = X.copy()
        for feature, name in zip(self.features, self.feature_names):

            source = pd.DataFrame(feature).set_index(name).values
            new_col = list()
            for x in range(len(target)):
                distances = []
                for y in range(len(source)):
                    distance = (
                        (source[y, 0] - target[x, 0]) ** 2
                        + (source[y, 1] - target[x, 1]) ** 2
                    ) ** 0.5
                    distances.append(distance)
                new_col.append(distances.index(min(distances)))
            X[name] = 0
            X[name] = new_col

        return X


def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d*1000 #distance in meters
