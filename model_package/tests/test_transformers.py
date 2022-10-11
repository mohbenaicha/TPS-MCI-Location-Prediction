import json
from pathlib import Path
from mci_model.config.base import config, DATASET_DIR
from mci_model.utilities import transformers as t
import pandas as pd


dt_features = config.model_config.datetime_features
eng_features = config.model_config.engineered_features
targets = config.model_config.targets


def test_datetime_imputer(test_input_data):
    test_input_data = pd.concat([test_input_data for _ in range(5)])
    # Given

    imputer = t.DateTimeImputer(dt_features['date'])
    
    assert test_input_data[[
        dt_features['day_of_year'],
        dt_features['month'],
        dt_features['day_of_week'],
        dt_features['day_of_month'],
        ]].iloc[10].to_list() == [14, 'January', 'Saturday  ', 14] 


    test_object = imputer.transform(test_input_data)

    # Then    
    assert test_object[[
        dt_features['day_of_year'],
        dt_features['month'],
        dt_features['day_of_week'],
        dt_features['day_of_month'],
        ]].iloc[10].to_list() == [14, 1, 5, 14]


def test_season_transformer(test_input_data):
    test_input_data = pd.concat([test_input_data for _ in range(5)])
    # Given
    imputer = t.DateTimeImputer(dt_features['date'])
    test_input_data = imputer.transform(test_input_data)
    
    transformer = t.SeasonTransformer(config.model_config.seasons)

    for idx, val in zip([22, 10, 50, 200], 
        [353, 14,226,161]):
        assert test_input_data[dt_features['day_of_year']].iat[idx] == val
    # assert test_input_data[dt_features['day_of_year']].iat[10] == 14
    # assert test_input_data[dt_features['day_of_year']].iat[50] == 226
    # assert test_input_data[dt_features['day_of_year']].iat[200] == 161

    test_object = transformer.fit_transform(test_input_data)

    # Then
    for idx, val in zip([22, 10, 50, 200], 
        ['Fall', 'Winter', 'Summer', 'Spring']):
        assert test_object[eng_features['season']].iat[idx] == val
    # assert test_object[eng_features.season].iat[10] == "Winter"
    # assert test_object[eng_features.season].iat[50] == "Summer"
    # assert test_object[eng_features.season].iat[200] == "Spring"
    assert len(test_object[eng_features['season']].unique()) == 4


def test_weekday_transformer(test_input_data):
    test_input_data = pd.concat([test_input_data for _ in range(5)])
    # Given
    imputer = t.DateTimeImputer(dt_features['date'])
    test_input_data = imputer.transform(test_input_data)

    transformer = t.WeekdayTransformer()

    assert test_input_data[dt_features['day_of_week']].iat[157] == 1 # Saturday
    assert test_input_data[dt_features['day_of_week']].iat[410] == 0 # Wednesday
    test_object = transformer.fit_transform(test_input_data)
    
    assert test_object[eng_features['weekday']].iat[157] == 'weekday'
    assert test_object[eng_features['weekday']].iat[410] == 'weekend'
    assert len(test_object[eng_features['weekday']].unique()) == 2


def test_tod_transformer(test_input_data):
    test_input_data = pd.concat([test_input_data for _ in range(5)])
    # Given
    transformer = t.ToDTransformer(config.model_config.levels)

    assert test_input_data[dt_features['hour']].iat[6] == 19
    assert test_input_data[dt_features['hour']].iat[10] == 13
    assert test_input_data[dt_features['hour']].iat[114] == 7
    
    test_object = transformer.fit_transform(test_input_data)

    # Then
    assert test_object[eng_features['ToD']].iat[6] == 'high'
    assert test_object[eng_features['ToD']].iat[10] == 'med'
    assert test_object[eng_features['ToD']].iat[114] == 'low'


def test_holiday_transformer(test_input_data):
    test_input_data = pd.concat([test_input_data for _ in range(5)])
    # Given
    transformer = t.HolidayTransformer(config.model_config.holidays)

    # and given Christmas day has corresponding report id's:
    
    test_object = transformer.fit_transform(test_input_data)

    idx = test_input_data[test_input_data[dt_features.get('day_of_year')] == 359].index
    assert all(test_input_data[test_input_data.index.isin(idx)][dt_features['day_of_year']] == 359)
    
    # Then
    assert all(test_object[test_object[dt_features['day_of_year']] == 359][
                eng_features['holiday']] == 'holiday')
    assert all(test_object[test_object[dt_features['day_of_year']] == 350][
                eng_features['holiday']] == 'non-holiday')



def test_new_feature_transformer(test_input_data):
    test_input_data = pd.concat([test_input_data for _ in range(1)])
    
    added_feature_names = [eng_features['pub'],
    eng_features['park'],
    eng_features['police_station']]
    added_features = list()
    for name in added_feature_names:
        with open(Path(f"{DATASET_DIR}/{name}.json"), 'rb') as f:
            features = json.load(f)
            added_features.append(features)
            f.close()

    transformer = t.NewFeatureTransformer(added_features, added_feature_names)
    test_object = transformer.fit_transform(X=test_input_data, y= test_input_data[[targets[0], targets[1]]])
    
    assert list(test_object[added_feature_names].iloc[0]) == [162, 61, 18]
    assert all([len((test_object.PS_Id.unique())) == 23, 
        len(test_object.Pub_Id.unique()) == 394,
        len(test_object.Park_Id.unique()) == 92])

