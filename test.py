import time
from datetime import datetime

# from src.utils.config import load_config 
from src.utils.time import convert_seconds

start = datetime.now()
a = sum(a**2 for a in range(1,10000000))

time = (datetime.now() - start).total_seconds()
print(convert_seconds(time))

# config = load_config("configs/data.yaml")

# idx_temp = [config['feature_cols'].index(c) for c in config['local_temporal_cols']]
# print(idx_temp)

