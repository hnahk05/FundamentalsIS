from pathlib import Path

import pandas as pd

from ..config import load_config, resolve_path


def aggregate_all_cables(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    constructed_dir = resolve_path(base, cfg["paths"]["data_constructed"])
    aggregated_dir = resolve_path(base, cfg["paths"]["data_aggregated"])
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    agg = cfg["aggregator"]
    x_columns = agg["x_columns"]
    y_column = agg["y_column"]
    x_axis = agg["x_axis"]
    x_window = agg["x_rolling_window"]
    y_window = agg["y_rolling_window"]

    files = [f for f in constructed_dir.iterdir() if "EXPORT" in f.name]

    for file_path in sorted(files):
        df = pd.read_csv(file_path)
        df_x_mean = df.loc[:, x_columns].rolling(x_window).mean()
        df_y_max = df.loc[:, y_column].rolling(y_window, center=True).max()

        df_overall = pd.merge(df_x_mean, df_y_max, left_index=True, right_index=True)
        df_overall = pd.merge(df_overall, df[x_axis], left_index=True, right_index=True)
        df_overall = df_overall.dropna()
        df_overall.to_csv(aggregated_dir / file_path.name, index=False)
        print(f"Aggregated: {file_path.name} -> {df_overall.shape}")
