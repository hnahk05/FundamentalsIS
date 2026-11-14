from pathlib import Path

import pandas as pd

from ..config import load_config, resolve_path


def _find_largest_monotonic_range(df: pd.DataFrame, kp_col: str) -> tuple[int, int]:
    max_length = 0
    best_range = (0, 0)
    i = 0
    while i < len(df) - 1:
        start = i
        end = i
        if df[kp_col].iloc[i] < df[kp_col].iloc[i + 1]:
            direction = "increasing"
        elif df[kp_col].iloc[i] > df[kp_col].iloc[i + 1]:
            direction = "decreasing"
        else:
            i += 1
            continue

        if direction:
            for j in range(i + 1, len(df) - 1):
                if (direction == "increasing" and df[kp_col].iloc[j] < df[kp_col].iloc[j + 1]) or (
                    direction == "decreasing" and df[kp_col].iloc[j] > df[kp_col].iloc[j + 1]
                ):
                    end = j + 1
                else:
                    break

        if end - start > max_length:
            max_length = end - start
            best_range = (start, end)

        i = end + 1

    return best_range


def clean_cable(file_path: Path, kp_col: str, prefix_to_strip: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    start, end = _find_largest_monotonic_range(df, kp_col)
    df_filtered = df.iloc[start:end]
    df_filtered.columns = df_filtered.columns.str.replace(prefix_to_strip, "")
    return df_filtered


def clean_all_cables(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    data_dir = resolve_path(base, cfg["paths"]["data"])
    cleaned_dir = resolve_path(base, cfg["paths"]["data_cleaned"])
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    kp_col = cfg["cleaner"]["kp_column_raw"]
    prefix = cfg["cleaner"]["prefix_to_strip"]

    files = [f for f in data_dir.iterdir() if "EXPORT" in f.name]
    for file_path in sorted(files):
        df = clean_cable(file_path, kp_col, prefix)
        df.to_csv(cleaned_dir / file_path.name, index=False)
        print(f"Cleaned: {file_path.name} -> {df.shape}")
