"""
DATA LOADER block > mage_0_ingest_data.py
"""

## DATA LOADER block > mage_0_ingest_data.py
if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

import io
import zipfile
from typing import List

import pandas as pd
import requests
import pyarrow.parquet as pq


@data_loader
def load_data(*args, **kwargs) -> pd.DataFrame:
    """
    Template for loading data from API

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Fetch the file using requests
    response = requests.get(
        f"https://archive.ics.uci.edu/static/public/848/secondary+mushroom+dataset.zip"
    )
    if response.status_code != 200:
        raise Exception(f"Failed to fetch file from url, status code: {response.status_code}")

    # Define paths
    zip_file_path = "data/secondary+mushroom+dataset.zip"
    inner_zip_path = "MushroomDataset.zip"
    specific_file = "MushroomDataset/secondary_data.csv"
    extract_path = "data/"

    ## Step 1: Open the outer zip file
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_file:
        ## Step 2: Extract the inner zip file as a ZipExtFile
        with zip_file.open(inner_zip_path) as zip_ext_file:
            ## Step 3: Read the contents of the ZipExtFile into memory a bytes buffer
            zip_ext_file_bytes = io.BytesIO(zip_ext_file.read())
            ## Step 4: Create a ZipFile object from the in-memory contents of the inner zip file
            with zipfile.ZipFile(zip_ext_file_bytes, "r") as inner_zip_file:
                ## Step 5: Extract the specific file from the inner ZipFile with folder structure
                # inner_zip_file.extract(specific_file, path=extract_path)
                ## Step 5: Extract the specific file without folder structure
                with inner_zip_file.open(specific_file) as f_in:
                    ## Define the path to save the extracted file
                    # with open('data/secondary_data.csv', 'wb') as f_out:
                    #     f_out.write(f_in.read())

                    ## Use io.BytesIO to handle the content in memory a bytes buffer
                    with io.BytesIO(f_in.read()) as f_buffer:
                        ## Read the content into a pandas DataFrame
                        df = pd.read_csv(f_buffer, sep=";")
                        print(df.shape)
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
