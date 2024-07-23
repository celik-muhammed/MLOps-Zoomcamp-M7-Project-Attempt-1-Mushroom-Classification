"""
pycode/preprocess_data.py
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
from sklearn.model_selection import train_test_split

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


def dump_pickle(obj, file_path: str) -> None:
    """
    Save an object to the specified file path using joblib.

    Parameters:
    obj: The object to be saved.
    file_path (str): The file path where the object will be saved.
                     Example: f"{dest_path}/{filename}"
    """
    ## Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    ## Save the object to the specified file path
    with open(file_path, "wb") as f_out:
        return joblib.dump(obj, f_out)


def load_pickle(file_path: str) -> tuple:
    """
    Load a pre-trained model from the specified file path using joblib.

    Parameters:
    file_path (str): The file path from which to load the model.
    """
    with open(file_path, "rb") as f_in:
        return joblib.load(f_in)


def read_data(data_path):
    """
    Read data from a specified file path.
    """
    df = pd.read_parquet(data_path)
    return df


def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the dataframe to a Parquet file at the specified output path using given storage options.
    """
    ## Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ## Save dataframe to Parquet file
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression=None,
        index=False,
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


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.25, SEED: int = 42):
    """
    Split the data into train, validation (optional), and test sets
    with a (60%/20%/20%) split ratio.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the dataset.
    test_size : float, default=0.2
        The proportion of the data to be used as the test set.
    val_size : float or None, default=0.25
        The proportion of the training data to be used as the validation set.
        If None, no validation set is created.
    SEED : int, default=42
        The random seed for reproducibility.

    Returns:
    -------
    tuple
        Splits of the data:
        - If val_size is not None: ((train_df, y_train), (val_df, y_val), (test_df, y_test))
        - If val_size is None: ((train_df, y_train), (test_df, y_test))
    """
    ## Maps categorical labels to numerical IDs.
    label2ids = {"edible": 0, "poisonous": 1}
    ## Invert the dictionary to create ids2label
    # ids2label = {v: k for k, v in label2ids.items()}

    ## Extract features (X) and target (y)
    X = df.drop(columns=["class"])  # Features (exclude the target)
    y = df["class"].map(label2ids)  # Target variable

    ## Split the data into training and test sets
    full_train_df, test_df, y_full_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=SEED
    )

    ## If val_size is specified, split the training set into training and validation sets
    if val_size is not None:
        train_df, val_df, y_train, y_val = train_test_split(
            full_train_df,
            y_full_train,
            stratify=y_full_train,
            test_size=val_size,
            random_state=SEED,
        )
        return (train_df, y_train), (val_df, y_val), (test_df, y_test)

    ## If val_size is None, return only the training and test sets
    return (full_train_df, y_full_train), (test_df, y_test)


def run_data_preparation_pipeline(
    year: int = 2023, month: int = 8, data_path: str = "./data/secondary_data.csv"
) -> None:
    """
    The main data preparation pipeline
    """
    # 0. Read data save as parquet
    logging.info("input_file %s", data_path)
    df_raw = pd.read_csv(data_path, sep=";")

    output_path = f"./data/secondary_data_{year:04d}-{month:02d}.parquet"
    logging.info("output_file %s", output_path)
    save_to_parquet(df_raw, output_path)

    ## Parameters
    data_path = f"./data/secondary_data_{year:04d}-{month:02d}.parquet"
    logging.info("input_file %s", data_path)

    # 1. Read data
    df = read_data(data_path)
    print(f"shape : {df.shape}")

    # 2. Preprocess data
    df = preprocess_data(df, features)

    # 3. Split data into training, validation and test sets
    train, val, test = split_data(df, test_size=0.2, val_size=0.25)

    ## 4. Save DictVectorizer and datasets
    output_path = "output"

    dump_pickle(train, f"{output_path}/train.joblib")
    logging.info("%s/train.joblib", output_path)

    dump_pickle(val, f"{output_path}/val.joblib")
    logging.info("%s/val.joblib", output_path)

    dump_pickle(test, f"{output_path}/test.joblib")
    logging.info("%s/test.joblib", output_path)


if __name__ == "__main__":
    ## Parameters
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2023  # 2023
    month = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # 8
    data_path = sys.argv[3] if len(sys.argv) > 3 else "./data/secondary_data.csv"
    print(year, month, data_path)

    ## Runs the entire preparation pipeline
    run_data_preparation_pipeline(year, month, data_path)
