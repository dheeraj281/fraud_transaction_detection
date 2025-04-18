from typing import List
import joblib
import os
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from fraud_transaction_detection import __version__ as _version  # noqa: F401
from fraud_transaction_detection.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = data_frame.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
    if hasattr(config, "unused_fields"):
        data_frame = data_frame.drop(columns=config.model_config_.unused_fields, errors='ignore')
    return data_frame


def load_dataset(*, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    df = save_for_testing_unseen_data(df)
    fraud_df = df[df['isFraud'] == 1]
    non_fraud_df = df[df['isFraud'] == 0]
    non_fraud_sample = non_fraud_df.sample(n=len(fraud_df)*5, random_state=42) 
    balanced_df = pd.concat([fraud_df, non_fraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    print("training data shape...", balanced_df.shape)
    transformed = pre_pipeline_preparation(data_frame=balanced_df)
    return transformed

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline trained successfully!")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py", ".gitignore"]
    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

def get_training_data() -> pd.DataFrame:
    """
    Fetch the training data from the dataset directory.
    """
    data = load_dataset(file_name=config.app_config_.training_data_file)
    features = config.model_config_.features

    X_train, X_test, y_train, y_test = train_test_split(
        data[features],  # predictors
        data[config.model_config_.target],
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state,
    )
    print("X_train shape",X_train.shape)
    print("X_test shape",X_test.shape)      
    print("y_train shape",y_train.shape)
    print("y_test shape",y_test.shape)
    return X_train, X_test, y_train, y_test

def save_for_testing_unseen_data(df: pd.DataFrame) -> pd.DataFrame:
    test_rows_0 = df[df["isFraud"] == 0].sample(n=2, random_state=42)
    test_rows_1 = df[df["isFraud"] == 1].sample(n=2, random_state=42)
    test_rows = pd.concat([test_rows_0, test_rows_1])
    test_rows.to_csv(DATASET_DIR / "test_rows.csv", index=False)
    df = df.drop(index=test_rows.index)
    return df
    
