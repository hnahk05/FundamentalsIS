from pathlib import Path

import pandas as pd


def peak_definer(
    df: pd.DataFrame,
    load_column: str = "Vehicle_Cable_Guide-I_Loadcell-Value",
    windows: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if windows is None:
        windows = [
            {"size": 600, "center": True, "multiplier": 2.5},
            {"size": 1200, "center": True, "multiplier": 2.5},
            {"size": 30, "center": False, "multiplier": 3.0},
        ]

    is_peaks = []
    for w in windows:
        rolling_mean = df[load_column].rolling(window=w["size"], center=w["center"]).mean()
        rolling_std = df[load_column].rolling(window=w["size"], center=w["center"]).std()
        is_peak = (df[load_column] > (rolling_mean + w["multiplier"] * rolling_std)).astype(int)
        is_peaks.append(is_peak)

    df["Is_Peak"] = is_peaks[0] | is_peaks[1] | is_peaks[2]
    return df, is_peaks[0], is_peaks[1], is_peaks[2]


def plot_all_peaks(
    intervals_csv: str | Path,
    constructed_dir: str | Path,
    figures_dir: str | Path,
    load_column: str = "Vehicle_Cable_Guide-I_Loadcell-Value",
    kp_column: str = "NAV_DATA_KP",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    cables_df = pd.read_csv(intervals_csv)
    cables_df["Start UTC"] = pd.to_datetime(cables_df["Start UTC"])
    cables_df = cables_df.sort_values(by="Start UTC").reset_index()
    cables = cables_df["Cycle"].tolist()

    for i in range(0, len(cables), 8):
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        axes = axes.flatten()

        for j, ax in enumerate(axes[: min(8, len(cables) - i)]):
            file_index = i + j
            df_temp = pd.read_csv(Path(constructed_dir) / f"EXPORT {cables[file_index]}.csv")
            df_temp, _, _, _ = peak_definer(df_temp, load_column)

            ax.plot(
                df_temp[kp_column],
                df_temp[load_column],
                label="Load",
                color="blue",
                linewidth=0.9,
            )
            ax.scatter(
                df_temp[kp_column][df_temp["Is_Peak"] == 1],
                df_temp[load_column][df_temp["Is_Peak"] == 1],
                color="red",
                label="Peaks",
                s=7,
            )
            ax.set_xlabel("KP")
            ax.set_ylabel("Load (tonnes)")
            ax.set_title(f"{cables[file_index]} (Starting {cables_df.iloc[file_index]['Start UTC']})")
            ax.legend()
            if df_temp[kp_column].iloc[0] > df_temp[kp_column].iloc[-1]:
                ax.invert_xaxis()

        plt.tight_layout()
        plt.savefig(Path(figures_dir) / f"Load Peak Identifier {i}.png")
        plt.close()


def plot_single_cable_peak_detail(
    file_path: str | Path,
    figures_dir: str | Path,
    load_column: str = "Vehicle_Cable_Guide-I_Loadcell-Value",
    kp_column: str = "NAV_DATA_KP",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path)
    df, is_peak1, is_peak2, is_peak3 = peak_definer(df, load_column)

    unique_peak_1 = is_peak1 == 1
    unique_peak_2 = (is_peak2 == 1) & ~(is_peak1 == 1)
    unique_peak_3 = (is_peak3 == 1) & ~(is_peak1 == 1) & ~(is_peak2 == 1)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df[kp_column], df[load_column], label="Load", color="grey")
    ax.scatter(
        df[kp_column][unique_peak_1],
        df[load_column][unique_peak_1],
        label="Peaks Window 1",
        color="green",
        s=22,
    )
    ax.scatter(
        df[kp_column][unique_peak_2],
        df[load_column][unique_peak_2],
        label="Peaks Window 2 (Additional)",
        color="blue",
        s=22,
    )
    ax.scatter(
        df[kp_column][unique_peak_3],
        df[load_column][unique_peak_3],
        label="Peaks Window 3 (Additional)",
        color="purple",
        s=22,
    )

    if df[kp_column].iloc[0] > df[kp_column].iloc[-1]:
        ax.invert_xaxis()

    ax.set_xlabel("KP")
    ax.set_ylabel("Load (tonnes)")
    ax.legend()
    plt.savefig(Path(figures_dir) / "peak_detail.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
