import pandas as pd

def load_and_split(data_path: str):
    """Load merged_dataset.csv and split by time. Return (train, val, test) DataFrames."""
    df = pd.read_csv(data_path, dtype={"location_id": str})
    # Normalize column names (e.g. if CSV uses "temperature_2m_max (°C)" or "DMI EAST")
    col_map = {}
    for c in list(df.columns):
        c_lower = c.strip().lower()
        if "temperature" in c_lower and "max" in c_lower and c != "max_temperature":
            col_map[c] = "max_temperature"
        if "dmi" in c_lower and "east" in c_lower and c != "dmi east":
            col_map[c] = "dmi east"
        if "nino" in c_lower and "3.4" in c_lower and c != "nino anom 3.4":
            col_map[c] = "nino anom 3.4"
    df = df.rename(columns=col_map)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["location_id", "time"]).reset_index(drop=True)

    train = df[(df["time"] >= "2005-01-01") & (df["time"] <= "2018-12-31")]
    val   = df[(df["time"] >= "2019-01-01") & (df["time"] <= "2021-12-31")]
    test  = df[(df["time"] >= "2022-01-01") & (df["time"] <= "2025-05-01")]

    return train, val, test

## Create a boolean fileter to keep only the true rows getting returned
def region_mask(df:pd.DataFrame, region_name: str):
    return df["region"].astype(str).str.strip().str.lower() == region_name.strip().lower()

def split_domain(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, source_domain: str, target_domain: str):
    source_train_df = train_df[region_mask(train_df, source_domain)]
    target_train_df = train_df[region_mask(train_df, target_domain)]
    target_val_df = val_df[region_mask(val_df, target_domain)]
    target_test_df = test_df[region_mask(test_df, target_domain)]

    return source_train_df, target_train_df, target_val_df, target_test_df

