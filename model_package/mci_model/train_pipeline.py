from config.base import config, DATASET_DIR
from pipeline import mci_pipeline
from sklearn.model_selection import train_test_split
from utilities.data_manager import load_dataset, save_pipeline

m_config = config.model_config
app_config = config.app_config

def train_model() -> None:
    data = load_dataset(file_name=f'{DATASET_DIR}/'+app_config.training_data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        data[m_config.train_features + m_config.inference_features_to_add],
        data[m_config.targets],
        test_size=m_config.test_size,
        random_state=m_config.random_state,
    )

    # fit model
    mci_pipeline.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=mci_pipeline)


if __name__ == "__main__":
    train_model()
