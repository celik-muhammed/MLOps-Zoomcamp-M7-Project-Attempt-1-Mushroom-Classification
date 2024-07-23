"""
DATA EXPORTER block > mage_2_hpo.py
"""

## DATA EXPORTER block > mage_2_hpo.py
if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import os
import sys

# import s3fs
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin  # , space_eval
from hyperopt.pyll import scope  # , stochastic
from sklearn.compose import make_column_transformer
from sklearn.metrics import roc_auc_score  # , log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder  # , OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler


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
        # force_int_remainder_cols=False,
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


def convert_to_integer_if_whole(params: dict):
    """
    Converts numeric values in a dictionary to integers if they are whole numbers.

    Parameters:
    params_dict (dict): Dictionary containing hyperparameters or numeric values.

    Returns:
    dict: Dictionary with numeric values converted to integers where applicable.
    """

    def convert_to_float(value):
        ## Check if value is a float number
        try:
            return float(value)
        except:
            return value

    def convert_to_int(value):
        ## Check if value is a whole number
        return int(value) if pd.api.types.is_number(value) and (value % 1 == 0) else value

    for key, value in params.items():
        value = convert_to_float(value)
        value = convert_to_int(value)
        params[key] = value
    return params


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
    train_df, y_train = data[0]
    val_df, y_val = data[1]
    test_df, y_test = data[2]

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
        ## Train a model
        model = model_builder(train_df, params).fit(train_df, y_train)
        ## Predict probabilities on validation data
        y_val_prob = model.predict_proba(val_df)[:, 1]  # Probability of positive class
        ## Set Model Evaluation Metric
        val_roc_auc = roc_auc_score(y_val, y_val_prob)

        return {"loss": -val_roc_auc, "status": STATUS_OK}

    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
        rstate=np.random.default_rng(42),  # for reproducible results
    )
    best_params = convert_to_integer_if_whole(best_params)
    print("best_params:", best_params)

    return model_builder(train_df, best_params).fit(train_df, y_train)
