from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import load_config, resolve_path


def plot_rq2_boxplot(
    x: str = "NAV_String_Out-Burial_Depth_boulder_percentage_mean",
    y: str = "NAV_String_Out-Burial_Depth_diff",
    cat: str = "Is_Peak",
    config_path: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    desc_dir = resolve_path(base, cfg["paths"]["descriptive_analysis"])

    df = pd.read_csv(desc_dir / "southfork.csv", sep=",")

    df_peak = df[(df[cat] == 1) & (df[y] != 0)]
    df_nopeak = df[(df[cat] == 0) & (df[y] != 0)]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.boxplot([df_peak[y], df_nopeak[y]])
    ax.set_xticklabels(["Peak", "No Peak"])
    ax.set_title(f"Boxplot distribution of {x}")
    ax.set_ylabel(x)
    plt.show()
