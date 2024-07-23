"""
DATA EXPORTER block > mage_3_register_model_s3.py
"""

## DATA EXPORTER block > mage_3_register_model_s3.py
if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import os

import joblib
import mlflow

## MLflow settings
## Build or Connect Database Offline/Online 'sqlite:///mlruns.db', 'http://127.0.0.1:5000'
## inter-container connection 'http://host.docker.internal:5001'
MLFLOW_TRACKING_URI = "http://host.docker.internal:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

## Build or Connect mlflow experiment
mlflow.set_experiment("xgboost-best-models")
mlflow.autolog(disable=True)

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
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    print(data)

    model = data

    with mlflow.start_run() as run:
        # Log pipeline parameters
        for name, step in model.named_steps.items():
            if hasattr(step, "get_params"):
                params = step.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(f"{name}_{param_name}", param_value)

        ## Save the best model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            pyfunc_predict_fn="predict_proba",
            # signature=signature
        )
        ## Register the best model
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="best-model",
        )
        print(f"RUN_ID: {run.info.run_id}")
    print("OK")
