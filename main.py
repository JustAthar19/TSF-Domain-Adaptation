from datetime import datetime

from src.data.loading import load_and_split, region_mask
from src.data.stations import select_target_stations_papua_by_elevation

from src.experiments.spatial_scarce import run_target_station_experiment

from src.utils.config import load_config
from src.utils.dataloader import get_dataloader_kwargs
from src.utils.seed import set_seed
from src.utils.runtime import setup_runtime

from src.utils.results import print_summary_table

from src.utils.io import save_to_csv

def main():
    
    config = load_config("configs/data.yaml")
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # device = setup_runtime()
    # dataloader_kwargs = get_dataloader_kwargs()

    set_seed(42)

   
    print(60*"=")
    print("Load and Split Dataset")
    train, val, test = load_and_split("data/merged_dataset.csv")

    print(f"Train: {train['time'].min()} to {train['time'].max()}  rows: {len(train)}")
    print(f"Val:   {val['time'].min()} to {val['time'].max()}  rows: {len(val)}")
    print(f"Test:  {test['time'].min()} to {test['time'].max()}  rows: {len(test)}")

    # Domain dataframes (source=Java; target=Papua)
    train_java = train[region_mask(train, "java")]
    train_papua = train[region_mask(train, "papua")]
    val_papua = val[region_mask(val, "papua")]
    test_papua = test[region_mask(test, "papua")]

    print("\n" + "=" * 60)
    print("Select Station Based on Elevation (Papua target)")
    print("=" * 60)
    single_id, three_ids, stats_sorted = select_target_stations_papua_by_elevation(train_papua, config)
    
    print(f"Papua training stations: {len(stats_sorted)}")
    print("Selected station IDs (by elevation):")
    print(f"  Single-station (median elevation): {single_id}")
    if len(three_ids) == 3:
        print(f"  Three-station (low/median/high):   {three_ids[0]}, {three_ids[1]}, {three_ids[2]}")
    else:
        print(f"  Three-station (deduped):           {', '.join(three_ids)}")

    
    print("\n" + "=" * 60)
    print(" Single Station Target Experiment")
    print("=" * 60)

    results_rows = []
    res_single, results_rows = run_target_station_experiment(
        run_ts=run_ts,
        result_rows=results_rows,
        experiment_name="single-station",
        train_java_df=train_java,
        train_papua_df=train_papua,
        val_papua_df=val_papua,
        test_papua_df=test_papua,
        config=config,
        target_station_ids=[single_id],
        epochs=100,
        batch_size=256,
        lr=1e-3,
        patience=5,
    )

    
    print("\n" + "=" * 60)
    print(" Three Station Target Experiment")
    print("=" * 60)
    res_three, results_rows = run_target_station_experiment(
        run_ts=run_ts,
        result_rows=results_rows,
        experiment_name="three-station",
        train_java_df=train_java,
        train_papua_df=train_papua,
        val_papua_df=val_papua,
        test_papua_df=test_papua,
        config=config,
        target_station_ids=three_ids,
        epochs=100,
        batch_size=256,
        lr=1e-3,
        patience=5,
    )

    print_summary_table(res_single, res_three)


    save_to_csv(run_ts=run_ts,results_rows=results_rows)


if __name__ == "__main__":
    main()