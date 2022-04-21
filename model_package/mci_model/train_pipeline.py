from config.base import config
from pipeline import mci_pipeline
from sklearn.model_selection import train_test_split
from utilities.data_manager import load_dataset, save_pipeline


def train_model() -> None:
    data = load_dataset(file_name=config.app_config.training_data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.targets],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # fit model
    mci_pipeline.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=mci_pipeline)


if __name__ == "__main__":
    train_model()
