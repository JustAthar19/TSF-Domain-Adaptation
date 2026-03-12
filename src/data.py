import pandas as pd
from .config import * 

def load_dataset(path):
    df = pd.read_csv(path)
    df = df.sort_values(["location_id", "time"])
    return df

def split_dataset(df):
    train = df[(df["time"] >= TRAIN_START) * (df["time"]<=TRAIN_END)]
    val   = df[(df["time"] >= VAL_START) & (df["time"] <= VAL_END)]
    test  = df[(df["time"] >= TEST_START) & (df["time"] <= TEST_END)]
    return train, val, test

