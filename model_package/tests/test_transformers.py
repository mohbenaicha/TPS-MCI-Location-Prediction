from mci_model.config.base import config
from mci_model.utilities import transformers as t


def test_season_tarnsformer(test_input_data):
    # Given
    transformer = t.SeasonTransformer(config.model_config.seasons)

    assert test_input_data["occurrencedayofyear"].iat[22] == 92.0
    test_object = transformer.fit_transform(test_input_data)

    # Then
    assert test_object["Season"].iat[22] == "Spring"


def test_weekday_transformer(test_input_data):
    # Given
    transformer = t.WeekdayTransformer()

    assert test_input_data["occurrencedayofweek"].iat[0] == "Saturday"
    test_object = transformer.fit_transform(test_input_data)

    # Then
    assert test_object["Weekend"].iat[0] == "weekend"


def test_tod_transformer(test_input_data):
    # Given
    transformer = t.ToDTransformer(config.model_config.levels)

    assert test_input_data["occurrencehour"].iat[0] == 13
    test_object = transformer.fit_transform(test_input_data)

    # Then
    assert test_object["ToDCrimeLevel"].iat[0] == "med"


def test_holiday_transformer(test_input_data):
    # Given
    transformer = t.HolidayTransformer(config.model_config.holidays)

    assert test_input_data["occurrencedayofyear"].iat[22] == 92.0
    test_object = transformer.fit_transform(test_input_data)

    # Then
    assert test_object["Holiday"].iat[22] == "holiday"
