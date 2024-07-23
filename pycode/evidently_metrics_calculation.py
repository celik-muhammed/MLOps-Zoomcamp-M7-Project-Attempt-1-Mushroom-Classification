## pycode/evidently_metrics_calculation.py
import io
import os
import time
import uuid
import random
import logging
import smtplib
import datetime
import warnings
from email.mime.text import MIMEText

import pytz
import numpy as np
import joblib
import pandas as pd

## Simple Scripts or Low-Level Access to PostgreSQL
# import psycopg2  # old
import psycopg  # (often referred to as psycopg3)

## High-Level Abstraction, high-level ORM (Object-Relational Mapping) to PostgreSQL
# from sqlalchemy import create_engine

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    ColumnQuantileMetric,
    DatasetMissingValuesMetric,
)
from evidently.metric_preset import ClassificationPreset

# from prefect import task, flow
# Set the timezone manually
os.environ["TZ"] = "UTC"
import pendulum

pendulum.now().in_tz("UTC")

# Suppress FutureWarning related to 'H' being deprecated
warnings.filterwarnings("ignore", category=FutureWarning, module="evidently")


def load_pickle(file_path: str) -> tuple:
    """
    Load a pre-trained model from the specified path.
    """
    with open(file_path, "rb") as f_in:
        return joblib.load(f_in)


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


def create_timestamp_feature(df, start_date="2023-08-01", end_date="2023-08-31", num_parts=8):
    """
    Adds a timestamp feature to the DataFrame and divides the data into specified parts based on the timestamp.

    Parameters:
    - df (pd.DataFrame): The DataFrame to which the timestamp feature will be added. It should have 60,000 rows.
    - start_date (str): The start date for generating timestamps. Defaults to '2023-08-01'.
    - end_date (str): The end date for generating timestamps. Defaults to '2023-08-31'.
    - num_parts (int): The number of parts to divide the month into. Defaults to 8.

    Returns:
    - pd.DataFrame: The DataFrame with two new columns:
        - 'timestamp': The generated timestamp for each row.
        - 'part': The integer index of the part each timestamp falls into.
    """
    ## Ensure df is a copy to avoid SettingWithCopyWarning
    df = df.copy()

    ## Generate a range of timestamps for the specified date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")  # Hourly frequency
    total_hours = len(date_range)

    ## Calculate how many timestamps to assign per part
    hours_per_part = total_hours // num_parts

    ## Create a range of timestamps for the DataFrame
    timestamps = pd.date_range(start=start_date, end=end_date, periods=len(df))
    df.loc[:, "timestamp"] = timestamps

    ## Function to assign each timestamp to one of the parts
    def assign_part(timestamp):
        ## Calculate the number of days since the start date
        days_since_start = (timestamp - pd.Timestamp(start_date)).days
        ## Determine the part index
        part_index = days_since_start * 24 // hours_per_part
        return min(
            part_index, num_parts - 1
        )  # Ensure part_index does not exceed the number of parts

    ## Apply the function to assign each timestamp to a part
    # df.loc[:, 'part'] = df['timestamp'].apply(assign_part)

    return df


# @task
def prep_db_connection(
    conn_string="user=root password=root host=host.docker.internal port=5432 dbname=postgres",
    test_db_conn_string=None,
):
    """
    Prepares a PostgreSQL database connection, creates a test database if it doesn't exist,
    and sets up a table named 'dummy_metrics'.

    Parameters:
    conn_string (str, optional): The connection string for the PostgreSQL database.
                                 Default is 'user=root password=root host=host.docker.internal port=5432 dbname=postgres'.
    test_db_conn_string (str, optional): The connection string for the 'test' database.
                                         If None, it is derived from the main connection string by replacing the database name.

    Returns:
    str: The connection string used for the 'test' database.
    """
    create_table_statement = """
    DROP TABLE IF EXISTS dummy_metrics;
    CREATE TABLE dummy_metrics (
        timestamp TIMESTAMP,
        prediction_drift FLOAT,
        num_drifted_columns INTEGER,
        share_missing_values FLOAT,
        cls_preset_f1 FLOAT
    );
    """

    # Establish the connection using the provided connection string
    with psycopg.connect(conn_string, autocommit=True) as conn:
        # Check if the 'test' database exists
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            # Create the 'test' database if it does not exist
            conn.execute("CREATE DATABASE test;")

    # If test_db_conn_string is not provided, derive it from the main connection string
    if test_db_conn_string is None:
        test_db_conn_string = conn_string.replace("dbname=postgres", "dbname=test")

    # Establish a new connection to the 'test' database
    with psycopg.connect(test_db_conn_string) as conn:
        # Create the 'dummy_metrics' table
        conn.execute(create_table_statement)

    # Return the connection string used for the 'test' database
    return test_db_conn_string


def monitor_metrics(df, reference_df):
    """
    Run the Evidently report for monitoring metrics.
    report (evidently.report.Report): The report object to store the metrics.
    column_mapping (ColumnMapping): Column mapping for metric calculations.
    """
    num_features = ["cap-diameter", "stem-height", "stem-width"]
    cat_features = [
        "class",
        "cap-shape",
        "cap-color",
        "does-bruise-or-bleed",
        "gill-color",
        "stem-color",
        "habitat",
        "season",
    ]
    column_mapping = ColumnMapping(
        task="classification",
        target="class",  #'y' is the name of the column with the target function
        pos_label=1,  # 'poisonous'
        prediction="prediction",  #'pred' is the name of the column(s) with model predictions
        numerical_features=num_features,  # list of numerical features
        categorical_features=cat_features,  # list of categorical features
        id=None,  # there is no ID column in the dataset
        datetime="timestamp",  #'date' is the name of the column with datetime
    )
    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnQuantileMetric(column_name="stem-width", quantile=0.5),
            ClassificationPreset(),
        ]
    )
    report.run(reference_data=reference_df, current_data=df, column_mapping=column_mapping)
    return report


def check_thresholds(report, metrics_thresholds):
    """Check if the metrics violate any thresholds."""
    result = report.as_dict()
    violations = []

    # Check if any metric violates the threshold
    for metric, threshold in metrics_thresholds.items():
        current_value = result["metrics"][4]["result"]["current"].get(metric)
        if current_value is not None and current_value < threshold:
            violations.append((metric, current_value, threshold))
    return violations


def send_alert(metric, current_value, threshold, email):
    """Send an alert via email."""
    msg = MIMEText(
        f"Alert: {metric} dropped to {current_value}, below the threshold of {threshold}."
    )
    msg["Subject"] = f"Metric Alert: {metric}"
    msg["From"] = email
    msg["To"] = email

    try:
        with smtplib.SMTP("localhost") as server:
            server.sendmail(email, [email], msg.as_string())
        logging.info("Alert sent via email.")
    except Exception as e:
        logging.error(f"Failed to send alert email: {e}")


def handle_violations(violations, email, actions={}):
    """Handle metric violations."""
    for violation in violations:
        metric, current_value, threshold = violation
        send_alert(metric, current_value, threshold, email)

        # if actions.get('retrain_model'):
        #     retrain_model()
        # if actions.get('generate_debugging_dashboard'):
        #     generate_debugging_dashboard()
        # if actions.get('switch_model'):
        #     switch_model()


def metric_calculation_postgresql(curr, idx: int, model, df: pd.DataFrame, ref_df: pd.DataFrame):
    """
    Calculates metrics from raw data, runs a report, and stores the results in a PostgreSQL database.

    Parameters:
    model (sklearn.base.BaseEstimator): Trained model used for making predictions.
    curr (psycopg2.extensions.cursor): Cursor object for PostgreSQL database connection.
    idx (int): Index to determine the date range for filtering the raw data.
    df (pd.DataFrame): The main DataFrame. Default load from 'data/green_tripdata_2024-03.parquet'.
    ref_df (pd.DataFrame): The reference DataFrame data used in the report.

    Returns:
    None: The function inserts the calculated metrics into the PostgreSQL database.
    """
    ## Define the time range for current data
    current_date_range = datetime.datetime(2023, 8, 1) + datetime.timedelta(days=idx)
    next_date_range = datetime.timedelta(days=1) + current_date_range
    ## Filter data
    current_df = df.loc[df.timestamp.between(current_date_range, next_date_range, inclusive="left")]

    ## Make predictions and fill missing values
    current_df["prediction"] = model.predict(current_df)

    ## Monitor metrics
    report = monitor_metrics(current_df, ref_df)

    ## Extract metrics from the report
    result = report.as_dict()
    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"]["share_of_missing_values"]
    cls_preset_f1 = result["metrics"][4]["result"]["current"]["f1"]

    ## Insert metrics into the PostgreSQL database
    curr.execute(
        "INSERT INTO dummy_metrics (timestamp, prediction_drift, num_drifted_columns, share_missing_values, cls_preset_f1) VALUES (%s, %s, %s, %s, %s)",
        (
            current_date_range,
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            cls_preset_f1,
        ),
    )

    ## Check thresholds
    violations = check_thresholds(report, {"f1": 0.80})

    ## Handle violations
    if violations:
        handle_violations(violations, "your_email@example.com")


# @flow
def run_batch_monitoring_backfill(
    raw_data_path: str = "data/secondary_data_2023-08.parquet",
):
    """
    Runs batch monitoring backfill by preparing the database connection,
    and performing metric calculations for a specified number of days.

    Returns:
    None
    """
    ## Prepare the database connection
    test_db_conn_string = prep_db_connection()

    ## Check if the file exists in local
    model_path = os.getenv("MODEL_PATH", "model/model2.joblib")
    if not os.path.exists(model_path):
        model_path = "model.joblib"
    ## Load model
    model = load_pickle(model_path)

    ## Load the necessary data
    ref_df = pd.read_parquet("data/reference_data.parquet")

    df_raw = pd.read_parquet(raw_data_path)
    df = preprocess_data(df_raw)
    ## Add timestamp and part columns to the DataFrame
    df = create_timestamp_feature(df)
    ## Maps categorical labels to numerical IDs.
    label2ids = {"edible": 0, "poisonous": 1}
    df["class"] = df["class"].map(label2ids)

    ## Set send timeout from environment variable or default to 10 seconds
    SEND_TIMEOUT = int(os.getenv("SEND_TIMEOUT", 10))
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=SEND_TIMEOUT)
    ## Connect to the 'test' database with autocommit enabled
    with psycopg.connect(test_db_conn_string, autocommit=True) as conn:
        for idx in range(28):  # Loop through the range of indices (0 to 27)
            with conn.cursor() as curr:
                ## Perform metric calculation for the current index
                metric_calculation_postgresql(curr, idx, model, df, ref_df)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            ## Wait if the elapsed time is less than the send timeout
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            ## Update last_send to ensure it advances correctly
            while last_send < new_send:
                last_send += datetime.timedelta(seconds=SEND_TIMEOUT)

            ## Log the data sent message with leading zeros for idx
            logging.info("%02d day Data Sent.", idx + 1)


if __name__ == "__main__":
    run_batch_monitoring_backfill()
