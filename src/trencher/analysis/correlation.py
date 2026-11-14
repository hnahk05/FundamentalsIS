from pathlib import Path

import pandas as pd

from ..config import load_config, resolve_path


def _correlation_analysis(df: pd.DataFrame, target: str = "Vehicle_Cable_Guide-I_Loadcell-Value") -> pd.DataFrame:
    df = df.select_dtypes(include=["number"])
    correlation = df.corrwith(df[target]).reset_index()
    correlation.columns = ["Variable", "Correlation"]
    correlation = correlation[correlation["Variable"] != target]
    correlation["AbsCorrelation"] = correlation["Correlation"].abs()
    return correlation.sort_values(by="AbsCorrelation", ascending=False).drop("AbsCorrelation", axis=1)


def _aggregation_df(df: pd.DataFrame, target: str = "Vehicle_Cable_Guide-I_Loadcell-Value") -> pd.DataFrame:
    indexer_backward = pd.api.indexers.FixedForwardWindowIndexer(window_size=15)
    indexer_forward = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    df = df.select_dtypes(include=["number"])

    df_mean = df.drop(columns=[target]).rolling(window=indexer_backward).mean()
    df_std = df.drop(columns=[target]).rolling(window=indexer_backward).var()
    df_max = df.drop(columns=[target]).rolling(window=indexer_backward).max()
    df_min = df.drop(columns=[target]).rolling(window=indexer_backward).min()
    df_max_target = df[target].rolling(window=indexer_forward).max()

    result = pd.merge(df_mean, df_std, left_index=True, right_index=True, suffixes=("_mean", "_std"))
    result = pd.merge(result, df_max, left_index=True, right_index=True, suffixes=("", "_max"))
    result = pd.merge(result, df_min, left_index=True, right_index=True, suffixes=("", "_min"))
    result = pd.merge(result, df_max_target, left_index=True, right_index=True, suffixes=("", "_max_cableguideload"))
    return result


def correlation_analysis_all(
    aggregate: bool = True,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    constructed_dir = resolve_path(base, cfg["paths"]["data_constructed"])
    desc_dir = resolve_path(base, cfg["paths"]["descriptive_analysis"])
    desc_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg["cleaner"]["prefix_to_strip"]
    files = sorted(f for f in constructed_dir.iterdir() if "EXPORT" in f.name)

    df_overall = pd.DataFrame(columns=["Variable"])
    for file_path in files:
        print(file_path.name)
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.replace(prefix, "")

        if aggregate:
            df = _aggregation_df(df)

        sorted_corrs = _correlation_analysis(df)
        sorted_corrs.rename(columns={"Correlation": file_path.name}, inplace=True)
        df_overall = pd.merge(df_overall, sorted_corrs, on="Variable", how="outer")

    df_overall.to_csv(desc_dir / "correlationAnalysisAll.csv", index=False, sep=";", decimal=",")
    return df_overall
