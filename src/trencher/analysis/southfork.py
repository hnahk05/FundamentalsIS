from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import load_config, resolve_path


def plot_southfork_soil_layers(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    constructed_dir = resolve_path(base, cfg["paths"]["data_constructed"])
    data_dir = resolve_path(base, cfg["paths"]["data"])
    figures_dir = resolve_path(base, cfg["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set()

    df = pd.read_csv(constructed_dir / "EXPORT K10-K09.csv", sep=",")
    df_soil_ids = pd.read_csv(data_dir / "soil_layers_id.csv", sep=",").iloc[0]

    fig, ax = plt.subplots(figsize=(12, 14), nrows=3, ncols=1, sharex=True)

    for col in ["L01", "L02", "L03"]:
        df[col + "_efs"] = df[col] - df["L00"]
        sns.lineplot(x="NAV_DATA_KP", y=col + "_efs", data=df, ax=ax[0], label=col)

    sns.lineplot(x="NAV_DATA_KP", y="NAV_String_Out-Burial_Depth", data=df, ax=ax[0], label="ACG")
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.1, 1.0), ncol=1, fontsize=14)
    ax[0].set_ylabel("Elevation from Seabed (m)", fontsize=14)
    ax[0].set_ylim(top=0, bottom=-2)
    ax[0].tick_params(axis="both", which="major", labelsize=14)
    ax[0].set_title("Layer Top Boundaries & ACG Elevation Over KP", fontsize=14)

    ax[1].plot(
        df["NAV_DATA_KP"],
        df["Vehicle_Cable_Guide-I_Loadcell-Value"],
        label="Load",
        color="blue",
        linewidth=0.9,
    )
    ax[1].scatter(
        df["NAV_DATA_KP"][df["Is_Peak"] == 1],
        df["Vehicle_Cable_Guide-I_Loadcell-Value"][df["Is_Peak"] == 1],
        color="red",
        label="Peaks",
        s=7,
    )
    ax[1].set_ylabel("Load (tonnes)", fontsize=14)
    ax[1].set_title("ACG Load (Tonnes) and Load Peaks over KP", fontsize=14)
    ax[1].legend(fontsize=14)

    ax[2].plot(
        df["NAV_DATA_KP"],
        df["Vehicle_control-Avg_Track_Speed"],
        label="Track Speed",
        color="blue",
        linewidth=0.9,
    )
    ax[2].set_ylabel("Speed (m/hr)", fontsize=14)
    ax[2].set_title("Average Track Speed over KP (m/hr)", fontsize=14)
    ax[2].legend(fontsize=14)

    plt.xlabel("KP (km)", fontsize=14)
    plt.tight_layout()
    plt.show()
