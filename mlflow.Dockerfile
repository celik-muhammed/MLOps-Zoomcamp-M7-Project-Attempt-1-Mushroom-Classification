## mlflow.Dockerfile
## The official MLflow Docker image is available on GHCR
## GitHub Container Registry at https://ghcr.io/mlflow/mlflow
## Use the official Python image from the Docker Hub
FROM python:3.11-slim

## Install required packages
RUN pip install pip mlflow boto3 s3fs -U

## Environment variables for AWS and MLflow
ENV AWS_ACCESS_KEY_ID="test"
ENV AWS_SECRET_ACCESS_KEY="test"
ENV AWS_REGION="us-east-1"
ENV AWS_ENDPOINT_URL="http://localhost:4566"
## Set the tracking URI using an environment variable
ENV MLFLOW_TRACKING_URI="sqlite:///mlruns.db"
ENV MLFLOW_HOME="/app"

## Expose the port the MLflow server will run on
EXPOSE 5000

## Set the working directory
WORKDIR "/app"

## Command to run the MLflow server
CMD [ \
  "mlflow", "server", \
  "--host", "0.0.0.0", \
  "--port", "5000", \
  "--backend-store-uri", "sqlite:///mlruns.db", \
  "--default-artifact-root", "s3://mushroom-dataset/model/", \
  "--serve-artifacts" \
]
