"""
pycode/predict_batch.py
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
import pandas as pd
from sklearn.pipeline import Pipeline

# Ignore all UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

## Define some environment variables
features = [
    "class",
    "cap-diameter",
    "cap-shape",
    "cap-color",
    "does-bruise-or-bleed",
    "gill-color",
    "stem-height",
    "stem-width",
    "stem-color",
    "habitat",
    "season",
]

del os.environ["AWS_ENDPOINT_URL"]


def load_pickle(file_path: str) -> tuple:
    """
    Load a pre-trained model from the specified path.
    """
    with open(file_path, "rb") as f_in:
        return joblib.load(f_in)


def get_input_path(year, month):
    """
    Get input pattern from environment variable or use default
    """
    default_input_pattern = "./data/secondary_data_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    """
    Get output pattern from environment variable or use default
    """
    default_output_pattern = "./output/secondary_data_{year:04d}-{month:02d}.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def get_storage_options() -> dict:
    """
    Get S3 storage options based on the environment variable `AWS_ENDPOINT_URL`.

    Returns:
        dict: A dictionary with storage options for S3 if `AWS_ENDPOINT_URL` is set,
              otherwise an empty dictionary.
    """
    options = {}
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    if endpoint_url:
        options = {
            "storage_options": {
                "client_kwargs": {"endpoint_url": endpoint_url}  # Localstack endpoint
            }
        }
    return options


def read_data(data_path):
    """
    Read data from a specified file path.
    """
    df = pd.read_parquet(data_path, **get_storage_options())
    return df


def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the dataframe to a Parquet file at the specified output path using given storage options.
    """
    if not os.getenv("AWS_ENDPOINT_URL"):
        ## Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ## Save dataframe to Parquet file
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression=None,
        index=False,
        **get_storage_options(),
    )


def preprocess_data(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """
    Preprocess the data (e.g., handle missing values, scale features).
    """
    ## Mapping dictionary
    mapping_dict = {
        "cap-shape": {
            "b": "bell",
            "c": "conical",
            "x": "convex",
            "f": "flat",
            "s": "sunken",
            "p": "spherical",
            "o": "others",
        },
        "cap-surface": {
            "i": "fibrous",
            "g": "grooves",
            "y": "scaly",
            "s": "smooth",
            "h": "shiny",
            "l": "leathery",
            "k": "silky",
            "t": "sticky",
            "w": "wrinkled",
            "e": "fleshy",
            "d": "dry",
        },
        "cap-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
        },
        "does-bruise-or-bleed": {"t": "bruises-or-bleeding", "f": "no"},
        "gill-attachment": {
            "a": "adnate",
            "x": "adnexed",
            "d": "decurrent",
            "e": "free",
            "s": "sinuate",
            "p": "pores",
            "f": "none",
            "?": "unknown",
        },
        "gill-spacing": {"c": "close", "d": "distant", "f": "none"},
        "gill-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
            "f": "none",
        },
        "stem-root": {
            "b": "bulbous",
            "s": "swollen",
            "c": "club",
            "u": "cup",
            "e": "equal",
            "z": "rhizomorphs",
            "r": "rooted",
            "f": "none",
        },
        "stem-surface": {
            "i": "fibrous",
            "g": "grooves",
            "y": "scaly",
            "s": "smooth",
            "h": "shiny",
            "l": "leathery",
            "k": "silky",
            "t": "sticky",
            "w": "wrinkled",
            "e": "fleshy",
            "d": "dry",
            "f": "none",
        },
        "stem-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
            "f": "none",
        },
        "veil-type": {"p": "partial", "u": "universal"},
        "veil-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
            "f": "none",
        },
        "has-ring": {"t": "ring", "f": "none"},
        "ring-type": {
            "c": "cobwebby",
            "e": "evanescent",
            "r": "flaring",
            "g": "grooved",
            "l": "large",
            "p": "pendant",
            "s": "sheathing",
            "z": "zone",
            "y": "scaly",
            "m": "movable",
            "f": "none",
            "?": "unknown",
        },
        "spore-print-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
        },
        "habitat": {
            "g": "grasses",
            "l": "leaves",
            "m": "meadows",
            "p": "paths",
            "h": "heaths",
            "u": "urban",
            "w": "waste",
            "d": "woods",
        },
        "season": {"s": "spring", "u": "summer", "a": "autumn", "w": "winter"},
        "class": {"e": "edible", "p": "poisonous"},
    }
    df = df.copy()

    ## Define some environment variables
    if features is None:
        features = [
            "class",
            "cap-diameter",
            "cap-shape",
            "cap-color",
            "does-bruise-or-bleed",
            "gill-color",
            "stem-height",
            "stem-width",
            "stem-color",
            "habitat",
            "season",
        ]

    ## Apply the mapping using map for each column
    for column, mapping in mapping_dict.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)

    ## Replace "none" and "unknown" with NaN using mask, where
    # df = df.replace(["none", "unknown"], np.NaN)
    # df = df.where(~df.isin(["none", "unknown"]), np.NaN)
    df = df.mask(df.isin(["none", "unknown"]), np.NaN)

    ## Drop identified columns from the DataFrame
    df = df[features].copy()
    ## Drop rows with any remaining NaN values
    df = df.dropna()
    return df


def prepare_data(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Prepares the data by adding a unique mushroom ID based on year and month.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing mushroom data.
    year (int): The year to include in the mushroom ID.
    month (int): The month to include in the mushroom ID.

    Returns:
    pd.DataFrame: Processed DataFrame with added mushroom ID column.
    """
    df["mushroom_id"] = f"{year:04d}/{month:02d}_" + df.index.astype(str)
    return df


def make_prediction(df: pd.DataFrame, model: Pipeline) -> np.ndarray:
    """
    Use the loaded model to make predictions on the preprocessed data.
    """
    y_pred = model.predict_proba(df)[:, 1]  # mlflow pyfunc model predict_proba
    return y_pred


def save_to_prediction(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Save the data with predictions.
    """
    df_result = pd.DataFrame()
    df_result["mushroom_id"] = df["mushroom_id"]
    df_result["predicted_duration"] = y_pred
    return df_result


def run_prediction_pipeline(year, month) -> None:
    """
    Run the entire prediction pipeline: load model, read data, preprocess data,
    make predictions, and save results.
    """
    ## Parameters
    data_path = f"./data/secondary_data_{year:04d}-{month:02d}.parquet"
    output_path = f"./output/secondary_data_{year:04d}-{month:02d}.parquet"
    logging.info("input_file  : %s", data_path)
    logging.info("output_file : %s", output_path)

    ## Check if the file exists in local
    model_path = os.getenv("MODEL_PATH", "model/model2.joblib")
    if not os.path.exists(model_path):
        model_path = "model.joblib"

    ## 1. Load model
    model = load_pickle(model_path)

    ## 2. Read data
    df = read_data(data_path)
    print(f"shape : {df.shape}")

    ## 3. Preprocess data
    df = preprocess_data(df, features)
    ## 4. Prepare data
    df = prepare_data(df, year, month)
    ## 5. Make prediction
    y_pred = make_prediction(df, model)
    ## Print Prediction
    print("predicted mean:", y_pred.mean().round(4))

    ## 6. Save prediction to df
    df = save_to_prediction(df, y_pred)
    ## 7. Save results to Parquet
    save_to_parquet(df, output_path)


if __name__ == "__main__":
    ## Parameters
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2023  # 2023
    month = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # 8
    print(year, month)

    ## Runs the entire prediction pipeline
    run_prediction_pipeline(year, month)
