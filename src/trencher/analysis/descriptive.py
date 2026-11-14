from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import load_config, resolve_path


def plot_descriptive_analysis(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    data_dir = resolve_path(base, cfg["paths"]["data"])
    figures_dir = resolve_path(base, cfg["paths"]["figures"])
    desc_dir = resolve_path(base, cfg["paths"]["descriptive_analysis"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    desc_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg["cleaner"]["prefix_to_strip"]

    df_intervals = pd.read_csv(data_dir / "Kaskasi_intervals.csv", sep=",")
    df_intervals["Start UTC"] = pd.to_datetime(df_intervals["Start UTC"])
    df_intervals["End UTC"] = pd.to_datetime(df_intervals["End UTC"])
    df_intervals["diff_hours"] = (
        (df_intervals["End UTC"] - df_intervals["Start UTC"]).dt.total_seconds() / 3600
    )
    print(f'Average difference in hours: {df_intervals["diff_hours"].mean():.2f}')

    df_intervals = df_intervals.sort_values(by=["Start UTC"])
    fig, ax = plt.subplots(figsize=(15, 14))
    sns.barplot(x="diff_hours", y="Cycle", data=df_intervals, ax=ax, color="darkblue")
    ax.set_xlabel("Duration in hours", fontsize=14)
    ax.set_ylabel("Cable", fontsize=14)
    fig.savefig(desc_dir / "kaskasiCableDuration.png", bbox_inches="tight", pad_inches=0.1)
    df_intervals.drop(columns="diff_hours").to_csv(desc_dir / "kaskasiCableDuration.csv", index=False)
    plt.close()

    df_sensor = pd.read_csv(data_dir / "EXPORT OSS-K29.csv", sep=",")
    df_sensor["Timestamp"] = pd.to_datetime(df_sensor["Timestamp"])
    df_sensor.columns = df_sensor.columns.str.replace(prefix, "")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="Vehicle_control-Avg_Track_Speed", data=df_sensor, ax=ax, color="darkblue", linewidth=1.2)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Average Track Speed in m/h", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisSpeedOverTime.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    df_sensor["trackSpeedDiff"] = df_sensor["Vehicle_Control-Port_Track_Speed"] - df_sensor["Vehicle_Control-Stbd_Track_Speed"]
    sns.lineplot(x="Timestamp", y="trackSpeedDiff", data=df_sensor, ax=ax, color="darkblue", linewidth=1.2)
    avgDiff = df_sensor["trackSpeedDiff"].mean()
    ax.axhline(avgDiff, color="red", linestyle="--", label=f"Average difference: {avgDiff:.2f} m/h")
    ax.legend(fontsize=14)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("PORT - STBD Track Speed in Δm/h", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisSpeedDiffOverTime.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="NAV_DATA_KP", data=df_sensor, ax=ax, color="darkblue", linewidth=1.2)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("KP point in km", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisKPpointOverTime.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="Vehicle_Cable_Guide-I_Loadcell-Value", data=df_sensor, ax=ax, color="darkblue", linewidth=1.2)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Reverse Force on Cable Guide in Tonnes", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisLoadcellOverTime.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="NAV_String_Out-Burial_Depth", data=df_sensor, ax=ax, color="darkblue", linewidth=1.2)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Negative Depth under Ground (m)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisBurialDepthOverTime.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="Vehicle_Cable_Guide-P_Force_Setpoint", data=df_sensor, ax=ax, color="darkblue", linewidth=1.2)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Force Setpoint in Tonnes", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    plt.close()

    df_jet = pd.read_csv(data_dir / "EXPORT OSS-K29.csv", sep=",")
    df_jet["Timestamp"] = pd.to_datetime(df_jet["Timestamp"])
    df_jet.columns = df_jet.columns.str.replace(prefix, "")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="Vehicle_Jetting-Port_Water_Gallery_Pressure", data=df_jet, ax=ax, color="red", linewidth=1.2)
    sns.lineplot(x="Timestamp", y="Vehicle_Jetting-Stbd_Water_Gallery_Pressure", data=df_jet, ax=ax, color="green", linewidth=1.2)
    line1 = mlines.Line2D([], [], color="red", label="PORT Jetting Pressure", linewidth=1.2)
    line2 = mlines.Line2D([], [], color="green", label="STBD Jetting Pressure", linewidth=1.2)
    ax.legend(handles=[line1, line2], loc="upper left", fontsize=14)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Pressure (bar)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisAftJetters.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="Vehicle_Fwd_Jetting-Jet_Tool_Width", data=df_jet, ax=ax, color="purple", linewidth=1.3)
    sns.lineplot(x="Timestamp", y="Vehicle_Jetting-Jet_Tool_Width", data=df_jet, ax=ax, color="blue", linewidth=1.3)
    line1 = mlines.Line2D([], [], color="purple", label="FWD Jetting Tool Width", linewidth=1.2)
    line2 = mlines.Line2D([], [], color="blue", label="AFT Jetting Tool Width", linewidth=1.2)
    ax.legend(handles=[line1, line2], loc="upper left", fontsize=14)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Width (mm)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisJetterWidth.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    _plot_case_analyses(data_dir, figures_dir, prefix)


def _plot_case_analyses(data_dir: Path, figures_dir: Path, prefix: str) -> None:
    df_k11 = pd.read_csv(data_dir / "EXPORT K11-K10.csv", sep=",")
    df_k11["Timestamp"] = pd.to_datetime(df_k11["Timestamp"])
    df_k11_case3 = df_k11[(df_k11["Timestamp"] >= "2022-05-14 11:59:57") & (df_k11["Timestamp"] <= "2022-05-14 12:01:01")]
    df_k11_case3.columns = df_k11_case3.columns.str.replace(prefix, "")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="NAV_String_Out-Burial_Depth", data=df_k11_case3, ax=ax, color="red", linewidth=1.2)
    ax2 = ax.twinx()
    sns.lineplot(x="Timestamp", y="Vehicle_Cable_Guide-I_Loadcell-Value", data=df_k11_case3, ax=ax2, color="blue", linewidth=1.2)
    ax.legend(["Cable Guide Depth"], loc="upper left", fontsize=14)
    ax2.legend(["Cable Guide Load"], loc="upper right", fontsize=14)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Negative Depth under Ground (m)", fontsize=14)
    ax2.set_ylabel("Cable Guide Load (tonnes)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisCase3.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    df_k27 = pd.read_csv(data_dir / "EXPORT K27-K25.csv", sep=",")
    df_k27["Timestamp"] = pd.to_datetime(df_k27["Timestamp"])
    df_k27_case1 = df_k27[(df_k27["Timestamp"] >= "2022-06-12 13:59:00") & (df_k27["Timestamp"] <= "2022-06-12 14:00:30")]
    df_k27_case1.columns = df_k27_case1.columns.str.replace(prefix, "")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="Vehicle_control-Avg_Track_Speed", data=df_k27_case1, ax=ax, color="red", linewidth=1.2)
    ax2 = ax.twinx()
    sns.lineplot(x="Timestamp", y="Vehicle_Cable_Guide-I_Loadcell-Value", data=df_k27_case1, ax=ax2, color="blue", linewidth=1.2)
    ax.legend(["Average Track Speed"], loc="upper left", fontsize=14)
    ax2.legend(["Cable Guide Load"], loc="upper right", fontsize=14)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Speed (m/h)", fontsize=14)
    ax2.set_ylabel("Cable Guide Load (tonnes)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisCase1.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    df_k27_case1["Speed_Diff"] = df_k27_case1["Vehicle_control-Avg_Track_Speed"].diff(10)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="Speed_Diff", data=df_k27_case1, ax=ax, color="red", linewidth=1.2)
    ax2 = ax.twinx()
    sns.lineplot(x="Timestamp", y="Vehicle_Cable_Guide-I_Loadcell-Value", data=df_k27_case1, ax=ax2, color="blue", linewidth=1.2)
    ax.legend(["Average Track Acceleration"], loc="upper left", fontsize=14)
    ax2.legend(["Cable Guide Load"], loc="upper right", fontsize=14)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Average Track Acceleration in 10 Seconds (Δm/h)", fontsize=14)
    ax2.set_ylabel("Cable Guide Load (tonnes)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisCase1_2.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    df_k11_case2 = df_k11[(df_k11["Timestamp"] >= "2022-05-14 15:49:37") & (df_k11["Timestamp"] <= "2022-05-14 16:00:10")]
    df_k11_case2.columns = df_k11_case2.columns.str.replace(prefix, "")
    df_k11_case2["Speed_Diff"] = (df_k11_case2["Vehicle_Control-Stbd_Track_Speed"] - df_k11_case2["Vehicle_Control-Port_Track_Speed"]).diff().abs()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x="Timestamp", y="Speed_Diff", data=df_k11_case2, ax=ax, color="red", linewidth=1.2)
    ax2 = ax.twinx()
    sns.lineplot(x="Timestamp", y="Vehicle_Cable_Guide-I_Loadcell-Value", data=df_k11_case2, ax=ax2, color="blue", linewidth=1.2)
    ax.legend(["Average Track Acceleration"], loc="upper left", fontsize=14)
    ax2.legend(["Cable Guide Load"], loc="upper right", fontsize=14)
    ax.grid()
    ax.set_xlabel("Timestamp (UTC)", fontsize=14)
    ax.set_ylabel("Average Track Speed Between Port & Stbd Tracks (Δm/h)", fontsize=14)
    ax2.set_ylabel("Cable Guide Load (tonnes)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.savefig(figures_dir / "descriptiveAnalysisCase1_3.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
