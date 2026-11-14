from pathlib import Path

import pandas as pd

from ..config import load_config, resolve_path
from ..model.peak_definer import peak_definer


def _calculate_percentage(
    trenching_tool_DuG: float, l_DuG: float, other_layers_DuG: list[float]
) -> float:
    closest_layer_below_DuG = 999999
    for ol_DuG in other_layers_DuG:
        if ol_DuG > l_DuG and ol_DuG < closest_layer_below_DuG:
            closest_layer_below_DuG = ol_DuG

    if l_DuG > trenching_tool_DuG:
        return 0.0
    if trenching_tool_DuG > closest_layer_below_DuG:
        return (closest_layer_below_DuG - l_DuG) / trenching_tool_DuG
    return (
        (closest_layer_below_DuG - l_DuG) - (closest_layer_below_DuG - trenching_tool_DuG)
    ) / trenching_tool_DuG


def _build_soil_layer_data(data_dir: Path, soil_layers: list[str]) -> pd.DataFrame:
    soil_layer_data = pd.read_csv(data_dir / "soil_layers.csv")

    count = 0.0
    for col in soil_layers:
        soil_layer_data[col + "_DuW"] = soil_layer_data[col] * -1 + count
        count += 0.000001

    top_layers = soil_layer_data[[s + "_DuW" for s in soil_layers]].min(axis=1)
    for col in soil_layers:
        soil_layer_data[col + "_DuG"] = soil_layer_data[col + "_DuW"] - top_layers

    boulder_depth_data = pd.read_csv(data_dir / "boulder_risk_depth.csv")
    soil_layer_data = pd.merge(
        left=soil_layer_data,
        right=boulder_depth_data,
        how="left",
        left_on=["project", "cycle", "kp"],
        right_on=["project", "cycle", "kp"],
    )
    soil_layer_data["boulder_depth"] = soil_layer_data["boulder_depth"] * -1
    return soil_layer_data


def _merge_cable(
    file_path: Path,
    soil_layer_data: pd.DataFrame,
    instruments_depths: list[str],
    soil_layers: list[str],
    burial_depth_diff_window: int,
) -> pd.DataFrame:
    cable_name = file_path.name.split("EXPORT ")[1].split(".")[0]
    print(cable_name)

    sensor_data = pd.read_csv(file_path)
    soil_cable = (
        soil_layer_data[soil_layer_data["cycle"] == cable_name]
        .sort_values(by=["kp"], ascending=True)
        .reset_index(drop=True)
    )
    print(f"{soil_cable.shape[0]} original rows.")
    soil_cable = soil_cable.dropna(subset="kp")
    print(f"{soil_cable.shape[0]} rows left.")

    if sensor_data["NAV_DATA_KP"].iloc[0] < sensor_data["NAV_DATA_KP"].iloc[1]:
        direction = "increasing"
    else:
        direction = "decreasing"
    print(direction)

    sensor_data = sensor_data.sort_values(by=["NAV_DATA_KP"], ascending=True).reset_index(drop=True)

    if sensor_data["NAV_String_Out-Burial_Depth"].mean() > 0:
        print("greater than 0")
        sensor_data["NAV_String_Out-Burial_Depth"] = sensor_data["NAV_String_Out-Burial_Depth"] * -1

    sensor_data = pd.merge_asof(
        left=sensor_data,
        right=soil_cable,
        left_on="NAV_DATA_KP",
        right_on="kp",
        direction="nearest",
    )

    if direction == "decreasing":
        sensor_data = sensor_data.sort_values(by=["NAV_DATA_KP"], ascending=False).reset_index(drop=True)

    print("Calculating soil layer proportions...")
    for instrument in instruments_depths:
        for col in soil_layers:
            sensor_data[instrument + "_" + col + "_percentage"] = sensor_data.apply(
                lambda x: _calculate_percentage(
                    abs(x[instrument]),
                    x[col + "_DuG"],
                    [x[s + "_DuG"] for s in soil_layers if s != col],
                ),
                axis=1,
            ).round(4)

        sensor_data[instrument + "_boulder_percentage"] = (
            (sensor_data[instrument] - sensor_data["boulder_depth"]) / sensor_data[instrument]
        ).clip(lower=0, upper=1)

    sensor_data["NAV_String_Out-Burial_Depth_Diff"] = (
        sensor_data["NAV_String_Out-Burial_Depth"].diff().rolling(burial_depth_diff_window).mean()
    )
    sensor_data, _, _, _ = peak_definer(sensor_data)
    print("Done")
    return sensor_data


def construct_all_cables(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    data_dir = resolve_path(base, cfg["paths"]["data"])
    cleaned_dir = resolve_path(base, cfg["paths"]["data_cleaned"])
    constructed_dir = resolve_path(base, cfg["paths"]["data_constructed"])
    constructed_dir.mkdir(parents=True, exist_ok=True)

    constr = cfg["constructor"]
    soil_layers = constr["soil_layers"]
    instruments_depths = constr["instruments_depths"]
    must_contain = constr["file_filter"]["must_contain"]
    diff_window = constr["burial_depth_diff_window"]

    soil_layer_data = _build_soil_layer_data(data_dir, soil_layers)

    files = [f for f in cleaned_dir.iterdir() if "EXPORT" in f.name]
    if len(must_contain) > 1:
        files = [f for f in files if all(kw in f.name for kw in must_contain)]
    else:
        files = [f for f in files if any(kw in f.name for kw in must_contain)]

    for file_path in sorted(files):
        df = _merge_cable(file_path, soil_layer_data, instruments_depths, soil_layers, diff_window)
        df.to_csv(constructed_dir / file_path.name, index=False)
