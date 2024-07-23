"""
pycode/hpo_s3.py
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

import numpy as np
import joblib
import mlflow
import pandas as pd
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin  # , space_eval
from hyperopt.pyll import scope  # , stochastic
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
            (
                "classifier",
                XGBClassifier(random_state=42, seed=42).set_params(**params),
            ),
        ]
    )
    return model


def run_optimization_log_model(data_path: str, model_path: str, num_trials=50) -> None:
    """
    The main optimization pipeline
    """
    ## params
    # EXPERIMENT_NAME = "xgboost-train"
    HPO_EXPERIMENT_NAME = "xgboost-hyperopt"

    ## Load train and test Data and preprocessor
    train_df, y_train = load_pickle(f"{data_path}/train.joblib")
    val_df, y_val = load_pickle(f"{data_path}/val.joblib")
    # print(type(train_df), type(y_train))

    ## MLflow settings
    ## Build or Connect Database Offline/Online 'sqlite:///mlruns.db', 'http://127.0.0.1:5000'
    ## inter-container connection 'http://host.docker.internal:5001'
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    ## Build or Connect mlflow experiment
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)

    ## References: https://mlflow.org/docs/latest/tracking/tracking-api.html#id1
    ## before your training code to enable automatic logging of sklearn metrics, params, and models
    ## Train your model (autologging will automatically start a new run)
    ## e.g., model.fit(X_train, y_train)
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)

    ## Define params
    search_space = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 90, 120, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 1, 5, 1)),
        "learning_rate": hp.quniform("learning_rate", 1e-2, 0.4, 1e-2),
        "min_child_weight": hp.quniform("min_child_weight", 1, 2, 1e-2),
        "scale_pos_weight": hp.quniform("scale_pos_weight", 0.5, 1, 1e-2),
        "subsample": hp.quniform("subsample", 0.6, 0.8, 1e-2),
        "random_state": 42,
        "seed": 42,
    }

    def objective(params):
        ## Start a manual MLflow run
        with mlflow.start_run(nested=True) as run:
            # print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
            ## Optional: Set some information about Model
            mlflow.set_tag("developer_name", "muce")
            mlflow.set_tag("train_data", f"{data_path}/train.pkl")
            mlflow.set_tag("val_data", f"{data_path}/val.pkl")
            mlflow.set_tag("test_data", f"{data_path}/test.pkl")

            ## Log the model params to the tracking server
            mlflow.log_params(params)

            ## Train a model (autologging will automatically log parameters, metrics and the model)
            model = model_builder(train_df, params).fit(train_df, y_train)

            ## Predict probabilities on validation data
            y_val_prob = model.predict_proba(val_df)[:, 1]  # Probability of positive class

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
            ## Infer the signature of the inputs and outputs using validation data and predictions
            signature = mlflow.models.infer_signature(val_df.head(), y_val_prob[:5])
            ## Log the trained scikit-learn, XGBoost model to MLflow with the inferred signature
            # mlflow.xgboost.log_model(model[-1],artifact_path="xgboost-model",signature=signature)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn-model",
                pyfunc_predict_fn="predict_proba",
                signature=signature,
            )
            ## Set Model Evaluation Metric
            val_roc_auc = roc_auc_score(y_val, y_val_prob)
            mlflow.log_metric("val_roc_auc", val_roc_auc)
        return {"loss": -val_roc_auc, "status": STATUS_OK}

    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=np.random.default_rng(42),  # for reproducible results
    )
    print("best_params:", best_params)


if __name__ == "__main__":
    ## Parameters
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./output"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "./model"
    print(data_path, model_path)

    ## Runs the entire training pipeline
    run_optimization_log_model(data_path, model_path)
