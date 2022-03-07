# Import the modules
from argparse import ArgumentParser
from re import I
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def DataPipeline(df: pd.DataFrame, args: ArgumentParser) -> pd.DataFrame:
    # Start tracking
    start = perf_counter()

    # Check some information about data
    if args.informations:
        data_informations(df=df)

    # Filter data by location
    df = df[df["location"] == args.location]

    # Choose necessary columns and filter the data
    df = df[
        [
            "date",
            "total_cases",
            "new_cases",
            "total_deaths",
            "new_deaths",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred",
            "new_deaths_per_million",
            "new_cases_per_million",
        ]
    ]

    # Change the date column type object to datetime
    df = df.astype({"date": "datetime64[ns]"})

    # Sort the values by the date variables
    df = df.sort_values(by=["date"], ascending=True, ignore_index=True)

    # Set the date as index and compute moving average with window=7 for new_cases and new_deaths
    df = df.set_index("date")

    # Feature engineering
    df = feature_engineering(df)

    df = df["7_days_MA_new_cases"]

    # Fill nan values with zero
    df = df.fillna(0)

    # Return the created dataframe
    return df


def feature_engineering(df):
    """With this function, make some manipulations on the features and prepare the data for the model.
    Args:
        df (pd.DataFrame): The given dataframe.
    Returns:
        This function returns the modified dataframe.
    """
    # Mean a given number of previous periods in a time series
    df["7_days_MA_new_cases"] = df["new_cases"].rolling(7).mean()
    df["7_days_MA_new_deaths"] = df["new_deaths"].rolling(7).mean()
    df["7_days_MA_new_cases_per_million"] = (
        df["new_cases_per_million"].rolling(7).mean()
    )
    df["7_days_MA_new_deaths_per_million"] = (
        df["new_deaths_per_million"].rolling(7).mean()
    )
    return df


def normalize(df):
    """With this fuction, Transform features by scaling each feature to a given range.
    Default Range: (0, 1)
     Args:
        df (pd.DataFrame): The given dataframe.
    Returns:
        This function returns the normalized dataframe
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalize_data = scaler.fit_transform(np.array(df.new_deaths).reshape(-1, 1))
    normalize_data = normalize_data.reshape(-1)
    return normalize_data


def data_informations(df):
    """ """
    # Check columns and data types
    print("Columns and data types:")
    print(df.info())
