"""
TRANSFORMER block > mage_1_preprocess_data.py
"""

## TRANSFORMER block > mage_1_preprocess_data.py
if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    y = df["class"].map(label2ids).to_frame()  # Target variable

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


@transformer
def transform(df: pd.DataFrame, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        df: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    ## Specify your transformation logic here
    ## 1. Preprocess data
    df = preprocess_data(df, features)

    ## 2. Split data into training, validation and test sets
    train, val, test = split_data(df, test_size=0.2, val_size=0.25)
    return train, val, test


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
