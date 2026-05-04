from datetime import datetime

from src.data.loading import load_and_split, split_domain
from src.data.stations import select_target_stations_papua_by_elevation

from src.experiments.spatial_scarce import run_target_station_experiment
from src.experiments.quantity_scarce import run_low_quantity_tgt_experiment

from src.utils.config import load_config
from src.utils.dataloader import get_dataloader_kwargs
from src.utils.seed import set_seed
from src.utils.runtime import setup_runtime

from src.utils.results import print_summary_table

from src.utils.io import save_to_csv
from src.utils.time import convert_seconds

def main():
    data_path = "data/merged_dataset.csv"
    data_config = load_config("configs/data.yaml")
    train_config = load_config("configs/train.yaml")
    
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # device = setup_runtime()
    # dataloader_kwargs = get_dataloader_kwargs()
    training_start_time = datetime.now()
    set_seed(42)

    
    input_len = data_config['input_len']
    horizon = data_config['horizon']
    stride = data_config['stride']
    
    feature_cols = data_config['feature_cols']
    target_col = data_config['target_col']

    source_domain  = data_config['source_domain']
    target_domain = data_config['target_domain']

    device = train_config['device']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    lr = train_config['lr']


    n_repeats = train_config['n_repeats']
    low_resource_fracs = train_config['low_resources_fracs']

   
    print(60*"=")
    print("Load and Split Dataset")
    train, val, test = load_and_split(data_path=data_path)

    print(f"Train: {train['time'].min()} to {train['time'].max()}  rows: {len(train)}")
    print(f"Val:   {val['time'].min()} to {val['time'].max()}  rows: {len(val)}")
    print(f"Test:  {test['time'].min()} to {test['time'].max()}  rows: {len(test)}")

    # Splitting domain 
    source_train_df, target_train_df, target_val_df, target_test_df = split_domain(train_df=train, val_df=val, test_df=test, source_domain=source_domain, target_domain=target_domain)


    # print("\n" + "=" * 60)
    # print("Select Station Based on Elevation")
    # print("=" * 60)
    # single_id, three_ids, stats_sorted = select_target_stations_papua_by_elevation(target_train_df, target_col)
    
    # print(f"Papua training stations: {len(stats_sorted)}")
    # print("Selected station IDs (by elevation):")
    # print(f"  Single-station (median elevation): {single_id}")
    # if len(three_ids) == 3:
    #     print(f"  Three-station (low/median/high):   {three_ids[0]}, {three_ids[1]}, {three_ids[2]}")
    # else:
    #     print(f"  Three-station (deduped):           {', '.join(three_ids)}")

    
    # print("\n" + "=" * 60)
    # print(" Single Station Target Experiment")
    # print("=" * 60)

    results_rows = []
    # res_single, results_rows = run_target_station_experiment(
    #     run_ts=run_ts,
    #     result_rows=results_rows,
    #     experiment_name="single-station",
    #     source_train_df=source_train_df,
    #     target_train_df=target_train_df,
    #     target_val_df=target_val_df,
    #     target_test_df=target_test_df,
    #     target_station_ids=[single_id],
    #     input_len=input_len,
    #     horizon=horizon,
    #     stride=stride,
    #     feature_cols=feature_cols,
    #     target_col=target_col,
    #     epochs=epochs,
    #     lr=lr,
    #     batch_size=batch_size,
    #     device=device
    # )

    
    # print("\n" + "=" * 60)
    # print(" Three Station Target Experiment")
    # print("=" * 60)
    # res_three, results_rows = run_target_station_experiment(
    #     run_ts=run_ts,
    #     result_rows=results_rows,
    #     experiment_name="three-station",
    #     source_train_df=source_train_df,
    #     target_train_df=target_train_df,
    #     target_val_df=target_val_df,
    #     target_test_df=target_test_df,
    #     target_station_ids=three_ids,
    #     input_len=input_len,
    #     horizon=horizon,
    #     stride=stride,
    #     feature_cols=feature_cols,
    #     target_col=target_col,
    #     epochs=epochs,
    #     lr=lr,
    #     batch_size=batch_size,
    #     device=device
    # )

    # print_summary_table(res_single, res_three)

    # print("")
    # training_end_time = convert_seconds((datetime.now() - training_start_time).total_seconds())
    # print(f"Spatial Scarcity Took: {training_end_time}")

    print(60*"=")
    print("Data Quantity Scarcity Experiment")
    
    results_rows = run_low_quantity_tgt_experiment(
        run_ts=run_ts,
        result_rows=results_rows,
        experiment_name="low-resource",
        source_train_df=source_train_df,
        target_train_df=target_train_df,
        target_val_df=target_val_df,
        target_test_df=target_test_df,
        n_repeats=n_repeats,
        low_resource_fracs=low_resource_fracs,
        input_len=input_len,
        horizon=horizon,
        stride=stride,
        feature_cols=feature_cols,
        target_col=target_col,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )

    training_end_time = convert_seconds((datetime.now() - training_start_time).total_seconds())
    print(f"Quantity Scarcity took {training_end_time}")
    save_to_csv(run_ts=run_ts,results_rows=results_rows)

    
    

if __name__ == "__main__":
    main()