import numpy as np

def _fmt(x):
    return f"{x:7.3f}" if (x is not None and not np.isnan(x)) else " nan"


def print_summary_table(res_single: dict, res_three: dict):
    print("\nSpatial Scarcity Summary Table (Papua test MAE)")
    print("=" * 60)
    methods = [k for k,v in res_single.items()]
    experiments = [
        ("Single-Station", res_single),
        ("Three-Station", res_three)
    ]
    
    print("Experiment        | " + " | ".join([f"{m} MAE".ljust(15) for m in methods]))
    print("-" * 60)
    for exp_name, res in experiments:
        row = " | ".join([f"{_fmt(res[m]['mae'])}".ljust(15) for m in methods])
        print(f"{exp_name:<17}" + " | " + f"{row}")
    
    print("-" * 60)
    print("\n" + "-" * 60)

    print("Experiment        | " + " | ".join([f"{m} MSE".ljust(15) for m in methods]))
    print("-" * 60)
    for exp_name, res in experiments:
        row = " | ".join([f"{_fmt(res[m]['mse'])}".ljust(15) for m in methods])
        print(f"{exp_name:<17}" + " | " + f"{row}")


    print("-" * 60)
    print("\n" + "-" * 60)

    print("Experiment        | " + " | ".join([f"{m} RMSE".ljust(15) for m in methods]))
    print("-" * 60)
    for exp_name, res in experiments:
        row = " | ".join([f"{_fmt(res[m]['rmse'])}".ljust(15) for m in methods])
        print(f"{exp_name:<17}" + " | " + f"{row}")

    print("-" * 60)
