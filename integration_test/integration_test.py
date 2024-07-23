"""
Integration Test Module

This module contains integration tests for the prediction pipeline.
It tests various aspects of data reading, processing, and saving to Parquet.
"""

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
from datetime import datetime

import pandas as pd

## Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append('../../')

# Import the save_data function
from pycode.predict_batch_s3 import read_data, get_input_path, get_output_path, save_to_parquet

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

## Setting environment variables for file patterns based on year and month
os.environ["INPUT_FILE_PATTERN"] = (
    "s3://mushroom-dataset/in/secondary_data_{year:04d}-{month:02d}.parquet"
)
os.environ["OUTPUT_FILE_PATTERN"] = (
    "s3://mushroom-dataset/out/secondary_data_{year:04d}-{month:02d}.parquet"
)


def dt(hour, minute, second=0):
    """
    Helper function to create a datetime object for the given hour, minute, and second.
    All datetime objects are set to January 1, 2023.

    Args:
    hour (int): Hour part of the datetime.
    minute (int): Minute part of the datetime.
    second (int, optional): Second part of the datetime. Defaults to 0.

    Returns:
    datetime: Datetime object with the specified time.
    """
    return datetime(2023, 1, 1, hour, minute, second)


def create_test_data():
    """
    Create a test DataFrame with sample data.
    This function generates a test DataFrame for demonstration or testing purposes
    """
    ## Pretend it's data for January 2023
    sample_data_dict_raw = {
        "class": {0: "p", 1: "p", 54362: "e", 55774: "e", 55775: "e"},
        "cap-diameter": {0: 15.26, 1: 16.6, 54362: 6.98, 55774: 8.6, 55775: 8.98},
        "cap-shape": {0: "x", 1: "x", 54362: "s", 55774: "o", 55775: "s"},
        "cap-surface": {0: "g", 1: "g", 54362: "w", 55774: None, 55775: None},
        "cap-color": {0: "o", 1: "o", 54362: "y", 55774: "n", 55775: "n"},
        "does-bruise-or-bleed": {0: "f", 1: "f", 54362: "f", 55774: "f", 55775: "f"},
        "gill-attachment": {0: "e", 1: "e", 54362: "f", 55774: None, 55775: None},
        "gill-spacing": {0: None, 1: None, 54362: "f", 55774: None, 55775: None},
        "gill-color": {0: "w", 1: "w", 54362: "f", 55774: "w", 55775: "w"},
        "stem-height": {0: 16.95, 1: 17.99, 54362: 6.5, 55774: 4.07, 55775: 5.09},
        "stem-width": {0: 17.09, 1: 18.19, 54362: 13.99, 55774: 23.89, 55775: 21.33},
        "stem-root": {0: "s", 1: "s", 54362: None, 55774: None, 55775: None},
        "stem-surface": {0: "y", 1: "y", 54362: None, 55774: None, 55775: None},
        "stem-color": {0: "w", 1: "w", 54362: "y", 55774: "n", 55775: "n"},
        "veil-type": {0: "u", 1: "u", 54362: None, 55774: None, 55775: None},
        "veil-color": {0: "w", 1: "w", 54362: None, 55774: None, 55775: None},
        "has-ring": {0: "t", 1: "t", 54362: "f", 55774: "f", 55775: "f"},
        "ring-type": {0: "g", 1: "g", 54362: "f", 55774: "f", 55775: "f"},
        "spore-print-color": {0: None, 1: None, 54362: None, 55774: None, 55775: None},
        "habitat": {0: "d", 1: "d", 54362: "d", 55774: "d", 55775: "d"},
        "season": {0: "w", 1: "u", 54362: "a", 55774: "a", 55775: "a"},
    }
    ## Sample df from sample_data_dict_raw
    sample_df_raw = pd.DataFrame(sample_data_dict_raw)
    ## Assuming `input_file` and `options` are defined appropriately for localstack S3
    ## 's3://mushroom-dataset/in/secondary_data_2023-01.parquet'  # Use INPUT_FILE_PATTERN here
    input_file = get_input_path(2023, 1)
    ## Save dataframe to S3
    save_to_parquet(sample_df_raw, input_file)
    logging.info("File saved to %s", input_file)


if __name__ == "__main__":
    create_test_data()

    # Run the batch script
    os.system("python pycode/predict_batch_s3.py 2023 01")

    ## Define output path and read the result
    ## 's3://mushroom-dataset/out/secondary_data_2023-01.parquet'  # Use OUTPUT_FILE_PATTERN here
    output_file = get_output_path(2023, 1)
    df_output = read_data(output_file)

    # Calculate the sum of predicted durations
    sum_predicted_durations = df_output["predicted_duration"].sum()
    print(f"Sum of predicted durations: {sum_predicted_durations}")

    # Verify the result
    assert (
        abs(sum_predicted_durations - 2) < 1e-2
    ), "Test failed: The sum of predicted durations is incorrect."
