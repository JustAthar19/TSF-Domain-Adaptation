TARGET_COL = "max_temperature"
FEATURE_COLS = [
    "max_temperature",
    "latitude",
    "longitude",
    "elevation",
    "sin_doy",
    "cos_doy",
    "solar_declination",
    "dmi east",
    "nino anom 3.4",
]
INPUT_LEN = 14
HORIZON = 7
STRIDE = 2

DEVICE = "cpu"

TRAIN_START = "2005-01-01"
TRAIN_END   = "2018-12-31"

VAL_START = "2019-01-01"
VAL_END   = "2021-12-31"

TEST_START = "2022-01-01"
TEST_END   = "2025-05-01"