import time
from datetime import datetime

from src.utils.config import load_config 
from src.utils.time import convert_seconds

from scipy.stats import wilcoxon


# start = datetime.now()
# a = sum(a**2 for a in range(1,10000000))

# time = (datetime.now() - start).total_seconds()
# print(convert_seconds(time))

# config = load_config("configs/data.yaml")

# idx_temp = [config['feature_cols'].index(c) for c in config['local_temporal_cols']]
# print(idx_temp)


# import numpy as np 
# rs_base = np.random.RandomState(123 + int(14))

# print(rs_base)
# print(type(rs_base))

# print("=" * 30 + " ARIMA "+ "=" * 30)

group1 = [20, 23, 21, 25, 18, 17, 18, 24, 20, 24, 23, 19]
group2 = [24, 25, 21, 22, 23, 18, 17, 28, 24, 27, 21, 23]

wilcox = wilcoxon(group1, group2)
print(wilcox)


# train_config = load_config("configs/train.yaml")
# print(train_config['low_resources_fracs'])


header = (
        f"{'INPUT_LEN':>10}  {'%Target':>8}  {'nTarget':>8}  "
        f"{'ATTF MAE':>18}  {'ATTF MSE':>18}  {'ATTF RMSE':>18}  "
        f"{'DAF MAE':>18}  {'DAF MSE':>18}  {'DAF RMSE':>18}  "
        f"{'p_mae':>10}  {'p_mse':>10}  {'p_rmse':>10}"
    )

row = (
    f"18.3383"
)
print(header)
