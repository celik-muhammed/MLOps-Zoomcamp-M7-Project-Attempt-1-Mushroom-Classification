"""
pycode/register_model_s3.py
"""

#!/usr/bin/env python
# coding: utf-8

# from typing import Any
import os
import sys

# import s3fs
# import pickle
import logging
import warnings

import joblib
import mlflow

# import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import make_column_transformer
from sklearn.metrics import roc_auc_score  # , log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder  # , OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

# Ignore all UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Set MPLBACKEND to 'Agg'
os.environ["MPLBACKEND"] = "Agg"

## Setting Up Environment Variables
## Load AWS credentials from environment variables
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
## Set Docker environment variable, export DOCKER_ENV='1'
os.environ["DOCKER_ENV"] = "1"
## Set Localstack S3 endpoint based on Docker environment
os.environ["AWS_ENDPOINT_URL"] = (
    f"http://{'host.docker.internal' if os.getenv('DOCKER_ENV') else 'localhost'}:4566"
)
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566')


def load_pickle(file_path: str) -> tuple:
    """
    Load a pre-trained model from the specified file path using joblib.

    Parameters:
    file_path (str): The file path from which to load the model.
    """
    with open(file_path, "rb") as f_in:
        return joblib.load(f_in)


def model_builder(train_df: pd.DataFrame, params: dict = {}):
    """
    model_builder for XGBClassifier
    """
    ## Select categorical columns
    cat = train_df.select_dtypes("O").columns

    ## Create a preprocessor to handle categorical columns with one-hot encoding
    preprocessor = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        remainder="passthrough",
        force_int_remainder_cols=False,
    )

    ## Create a logistic regression model
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("classifier", XGBClassifier(**params)),
        ]
    )
    return model


def convert_to_integer_if_whole(params: dict) -> dict:
    """
    Converts numeric values in a dictionary to integers if they are whole numbers.

    Parameters:
    params (dict): Dictionary containing hyperparameters or numeric values.

    Returns:
    dict: Dictionary with numeric values converted to integers where applicable.
    """

    def convert_to_float(value):
        """
        Converts a value to a float if possible.

        Parameters:
        value: The value to be converted.

        Returns:
        float or original value: The converted float value or the original value if fails.
        """
        try:
            return float(value)
        except ValueError:
            return value

    def convert_to_int(value):
        """
        Converts a float value to an integer if it is a whole number.

        Parameters:
        value: The value to be checked and possibly converted.

        Returns:
        int or original value: The converted integer value
                               if it is a whole number or the original value.
        """
        return int(value) if pd.api.types.is_number(value) and (value % 1 == 0) else value

    for key, value in params.items():
        value = convert_to_float(value)
        value = convert_to_int(value)
        params[key] = value

    return params


def train_and_log_model(
    params: dict,
    experiment_name: str,
    data_path: str = "data",
    model_path: str = "model",
) -> None:
    """The main training pipeline"""
    ## params
    # EXPERIMENT_NAME = "xgboost-train"

    ## Load train and test Data and preprocessor
    train_df, y_train = load_pickle(f"{data_path}/train.joblib")
    val_df, y_val = load_pickle(f"{data_path}/val.joblib")
    test_df, y_test = load_pickle(f"{data_path}/test.joblib")
    # print(type(train_df), type(y_train))

    ## MLflow settings
    ## Build or Connect Database Offline/Online 'sqlite:///mlruns.db', 'http://127.0.0.1:5000'
    ## inter-container connection 'http://host.docker.internal:5001'
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    ## Build or Connect mlflow experiment
    mlflow.set_experiment(experiment_name)

    ## References: https://mlflow.org/docs/latest/tracking/tracking-api.html#id1
    ## before your training code to enable automatic logging of sklearn metrics, params, and models
    ## Train your model (autologging will automatically start a new run)
    ## e.g., model.fit(X_train, y_train)
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)

    ## Start a manual MLflow run
    with mlflow.start_run() as run:
        # print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
        ## Optional: Set some information about Model
        mlflow.set_tag("developer_name", "muce")
        mlflow.set_tag("train_data", f"{data_path}/train.pkl")
        mlflow.set_tag("val_data", f"{data_path}/val.pkl")
        mlflow.set_tag("test_data", f"{data_path}/test.pkl")

        ## Set Model params information
        ## Convert a string parameter back to its original type.
        params = convert_to_integer_if_whole(params)
        mlflow.log_params(params)

        ## Train a model (autologging will automatically log parameters, metrics, and the model)
        model = model_builder(train_df, params).fit(train_df, y_train)

        ## Predict probabilities on validation data
        y_val_prob = model.predict_proba(val_df)[:, 1]  # Probability of positive class
        y_test_prob = model.predict_proba(test_df)[:, 1]  # Probability of positive class

        ## Log a specific file or a directory and its contents recursively (if any)
        ## Create dest_path folder unless it already exists
        # pathlib.Path(dest_path).mkdir(exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        ## Define the file path
        file_path = os.path.join(model_path, "model2.joblib")  # f"{model_path}/{filename}"
        with open(file_path, "wb") as f_out:
            joblib.dump(model, f_out)
        ## To log a custom process like a preprocessor, model, or pipeline as a pickle file
        mlflow.log_artifact(local_path=file_path, artifact_path="pickle-model")

        ## (Optional) Log only model via sklearn (not Pipeline)
        ## Infer the signature of the model inputs and outputs using validation data and prediction
        signature = mlflow.models.infer_signature(val_df.head(), y_val_prob[:5])
        ## Log the trained scikit-learn, XGBoost model to MLflow with the inferred signature
        # mlflow.xgboost.log_model(model[-1], artifact_path="xgboost-model", signature=signature)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            pyfunc_predict_fn="predict_proba",
            signature=signature,
        )
        ## Set Model Evaluation Metric
        val_roc_auc = roc_auc_score(y_val, y_val_prob)
        mlflow.log_metric("val_roc_auc", val_roc_auc)

        ## Set Model Evaluation Metric
        test_roc_auc = roc_auc_score(y_test, y_test_prob)
        mlflow.log_metric("test_roc_auc", test_roc_auc)


def run_register_model(data_path: str, top_n: int) -> None:
    """
    The main register pipeline
    """
    ## Parameters
    EXPERIMENT_NAME = "xgboost-best-models"
    HPO_EXPERIMENT_NAME = "xgboost-hyperopt"
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # 'sqlite:///mlruns.db'

    ## Initialize the MLflow client with a specific tracking URI
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    ## Retrieve the top_n model by val metrics runs and log the models
    runs = client.search_runs(
        experiment_ids=client.get_experiment_by_name(HPO_EXPERIMENT_NAME).experiment_id,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=top_n,
        ## Search runs and order by validation loss in Ascending or increasing order
        ## Search runs and order by validation ROC AUC in Descending or decreasing order
        order_by=["metrics.val_roc_auc DESC"],
    )
    for run in runs:
        train_and_log_model(
            params=run.data.params,
            experiment_name=EXPERIMENT_NAME,
            data_path=data_path,
        )

    ## Select the model with the suitable test metrics
    best_run = client.search_runs(
        experiment_ids=client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1,
        ## Search runs and order by validation loss in Ascending or increasing order
        ## Search runs and order by validation ROC AUC in Descending or decreasing order
        order_by=["metrics.test_roc_auc DESC"],
    )[0]

    ## Register the best model
    mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/model",
        name="best-model",
    )
    print(f'Test Score of the best model: {best_run.data.metrics["test_roc_auc"]:.4f}')


if __name__ == "__main__":
    ## Parameters
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./output"
    top_n = sys.argv[2] if len(sys.argv) > 2 else 5
    print(data_path, top_n)

    ## Runs the entire training pipeline
    run_register_model(data_path, top_n)
