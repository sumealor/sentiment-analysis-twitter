
import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def load_data(path):
    """
    Loads the dataset from a CSV file.

    Args:
        path (str): File path to CSV
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    
    """
    return pd.read_csv(path)


def clean_data(df):
    """
    Cleans the dataset by removing leaky/uncessary columns and 
    extracting time-based features from tweet timestamps.
    
    Args:
        df(pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame : Cleaned dataframe
    """        
    # Remove features that leak information or are not useful 
    leaky = [ 
        "airline_sentiment_confidence", "negativereason", "negativereason_confidence",
        "airline_sentiment_gold", "negativereason_gold", "name", "tweet_id"
    ]

    df = df.drop(columns=leaky)
    
    # Extract datetime features
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')

    df['tweet_hour'] = df['tweet_created'].dt.hour
    df['tweet_dayofweek'] = df['tweet_created'].dt.dayofweek
    df['tweet_is_weekend'] = df['tweet_dayofweek'].isin([5, 6]).astype(int)


    df = df.drop(columns=['tweet_created',"tweet_coord"])
    return df
