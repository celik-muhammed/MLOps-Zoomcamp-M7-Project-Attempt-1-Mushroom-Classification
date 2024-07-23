"""
tests/test_batch_s3.py
"""

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
from pprint import pprint  # printed in multiple lines
from datetime import datetime

# import pytest
import pandas as pd

## Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append('../../')

# Import the save_data function
from pycode.predict_batch_s3 import features, preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


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


def test_prepare_data():
    """
    Test the prepare_data function to ensure it correctly processes and transforms input DataFrame.

    The test creates sample DataFrame with test data, processes it using the prepare_data function,
    and then compares the actual output with the expected output.
    """
    ## Sample raw data
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

    ## Process the DataFrame using the prepare_data function and features
    sample_df_raw = preprocess_data(sample_df_raw, features)

    ## Print the actual DataFrame
    print("Actual DataFrame:")
    pprint(sample_df_raw)

    # Define the expected output data after processing
    sample_data_dict_expected = {
        "class": {0: "poisonous", 1: "poisonous", 55774: "edible", 55775: "edible"},
        "cap-diameter": {0: 15.26, 1: 16.6, 55774: 8.6, 55775: 8.98},
        "cap-shape": {0: "convex", 1: "convex", 55774: "others", 55775: "sunken"},
        "cap-color": {0: "orange", 1: "orange", 55774: "brown", 55775: "brown"},
        "does-bruise-or-bleed": {0: "no", 1: "no", 55774: "no", 55775: "no"},
        "gill-color": {0: "white", 1: "white", 55774: "white", 55775: "white"},
        "stem-height": {0: 16.95, 1: 17.99, 55774: 4.07, 55775: 5.09},
        "stem-width": {0: 17.09, 1: 18.19, 55774: 23.89, 55775: 21.33},
        "stem-color": {0: "white", 1: "white", 55774: "brown", 55775: "brown"},
        "habitat": {0: "woods", 1: "woods", 55774: "woods", 55775: "woods"},
        "season": {0: "winter", 1: "summer", 55774: "autumn", 55775: "autumn"},
    }
    ## Sample df from sample_data_dict_expected
    sample_df_expected = pd.DataFrame(sample_data_dict_expected)

    ## Print the expected DataFrame
    print("\nExpected DataFrame:")
    pprint(sample_df_expected)

    # Use Pandas testing utility to compare the actual and expected DataFrames
    pd.testing.assert_frame_equal(sample_df_raw, sample_df_expected)


if __name__ == "__main__":
    # pytest.main()
    test_prepare_data()
