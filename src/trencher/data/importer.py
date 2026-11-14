import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from ..config import load_config, resolve_path


def _load_credentials(config_path: str | Path | None = None):
    for candidate in [
        Path("local_config.json"),
        Path("../local_config.json"),
        Path(__file__).resolve().parents[3] / "local_config.json",
    ]:
        if candidate.exists():
            with open(candidate) as fh:
                return json.load(fh)
    print(
        "local_config.json not found. "
        "Please create one with keys: osisoft_pi_username, osisoft_pi_password, D398"
    )
    sys.exit(1)


def _create_session(creds: dict) -> requests.Session:
    requests.packages.urllib3.disable_warnings(
        requests.packages.urllib3.exceptions.InsecureRequestWarning
    )
    session = requests.Session()
    session.auth = (creds["osisoft_pi_username"], creds["osisoft_pi_password"])
    return session


def _get_dataserver_metadata(source: dict, session: requests.Session) -> dict:
    url = source["serverurl"] + "piwebapi/dataservers/"
    response = session.get(url, verify=False)
    if response.status_code == 200:
        data = response.json()
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} -- dataservers:')
        for item in data["Items"]:
            print(f'\t\t\t\t\t-- WebId: {item["WebId"]}')
        return data
    print(f"status_code: {response.status_code}\n{response.text}")
    return {}


def _get_point_metadata(
    name_filter: str, source: dict, session: requests.Session, cache_path: Path | None = None
) -> dict:
    url = source["serverurl"] + "piwebapi/dataservers/" + source["dataserver"] + "/points"
    response = session.get(
        url,
        params={
            "nameFilter": name_filter,
            "maxCount": 100000,
            "selectedFields": "Items.Name;Items.Descriptor;Items.EngineeringUnits;"
            "Items.PointType;Items.WebId",
        },
        verify=False,
    )
    if response.status_code == 200:
        data = response.json()
        if cache_path:
            with open(cache_path, "w") as fh:
                json.dump(data, fh)
        return data
    print(f"status_code: {response.status_code}\n{response.text}")
    return {}


def _get_point_value(
    name: str,
    start: str,
    end: str,
    source: dict,
    parameters: dict,
    session: requests.Session,
    source_meta: dict,
) -> pd.DataFrame | None:
    df_total = pd.DataFrame(columns=["Timestamp", name])
    try:
        parameters["WebId"] = _get_point_metadata(name, source, session)["Items"][0]["WebId"]
    except (KeyError, IndexError):
        return None

    max_points = 100000
    if source["streamsets"] == "interpolation":
        freq = parameters.get("interval", "1s")
    elif source["streamsets"] == "recorded":
        freq = "500ms"
    else:
        freq = parameters.get("summaryDuration", "1s")

    total_points = int(pd.date_range(start=start, end=end, freq=freq, inclusive="left").shape[0])

    if total_points > max_points:
        dates = pd.Series(
            pd.date_range(
                start=start,
                end=end,
                periods=int(math.ceil(total_points / max_points)) + 1,
            )
        )
        intervals = pd.DataFrame({"start": dates.iloc[:-1].values, "end": dates.iloc[1:].values})
    else:
        intervals = pd.DataFrame({"start": [start], "end": [end]})

    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} -- '
        f"Max data points: {max_points:,} | Total data points: {total_points:,}"
    )

    for _, row in intervals.iterrows():
        url = source["serverurl"] + "piwebapi/streamsets/" + source["streamsets"]
        parameters["starttime"] = row.iloc[0]
        parameters["endtime"] = row.iloc[1]
        response = session.get(url, params=parameters, verify=False)

        if response.status_code == 200:
            data = response.json()
            if source["streamsets"] == "summary":
                df = pd.DataFrame(data["Items"][0]["Items"]).Value.apply(pd.Series)
            else:
                df = pd.DataFrame(data["Items"][0]["Items"]).apply(pd.Series)

            if df.shape[0] > 0:
                df = df[["Timestamp", "Value"]]
                df.rename(columns={"Value": name}, inplace=True)
                df_total = pd.concat([df_total, df])
                del df
        else:
            print(f"status_code: {response.status_code}\n{response.text}")
            return None

    return df_total


def import_sensor_data(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    data_dir = resolve_path(base, cfg["paths"]["data"])
    data_dir.mkdir(parents=True, exist_ok=True)

    creds = _load_credentials(config_path)
    session = _create_session(creds)

    imp = cfg["importer"]
    source = {
        "serverurl": imp["pi_server_url"],
        "dataserver": imp["pi_dataserver_webid"],
        "streamsets": imp["pi_streamsets"],
    }
    parameters = dict(imp["pi_parameters"])

    _get_dataserver_metadata(source, session)
    metadata = _get_point_metadata(
        imp["pi_name_filter"], source, session, cache_path=data_dir / "imported_tags.json"
    )

    intervals = pd.read_csv(data_dir / imp["intervals_csv"])

    for _, row in intervals.iterrows():
        cable = row["Cycle"]
        print(cable)
        start = row["Start UTC"]
        end = row["End UTC"]

        if cable in "".join(os.listdir(data_dir)):
            print("Already exists, skipping...")
            continue

        dfs = []
        for item in tqdm(metadata["Items"]):
            df_temp = _get_point_value(item["Name"], start, end, source, parameters, session, metadata)
            if df_temp is not None and not df_temp.empty:
                df_temp.set_index("Timestamp", inplace=True, drop=True)
                dfs.append(df_temp)

        print("Merging...")
        df = pd.concat(dfs, axis=1, join="outer")
        df.reset_index(inplace=True, drop=False)
        df.to_csv(data_dir / f"EXPORT {cable}.csv", index=False)


def import_postgres_tables(config_path: str | Path | None = None) -> None:
    import logging

    from sqlalchemy import create_engine

    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    data_dir = resolve_path(base, cfg["paths"]["data"])
    data_dir.mkdir(parents=True, exist_ok=True)

    creds = _load_credentials(config_path)
    cfg_d398 = creds["D398"]
    engine = create_engine(
        f'postgresql://{cfg_d398["user"]}:{cfg_d398["password"]}@'
        f'{cfg_d398["host"]}:{cfg_d398["port"]}/{cfg_d398["database"]}'
    )

    pg = cfg["importer"]["postgres"]

    for table in pg["tables"]["small"]:
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        df.to_csv(data_dir / f"{table}.csv", index=False)

    if pg["tables"].get("wee_bankie_depth"):
        df = pd.read_sql("SELECT * FROM wee_bankie_depth", engine)
        df.to_csv(data_dir / "wee_bankie_depth.csv", index=False)

    if "trencher" in pg["tables"]:
        t_cfg = pg["tables"]["trencher"]
        offset = t_cfg["offset"]
        limit = t_cfg["limit"]
        while True:
            sql = f"SELECT * FROM trencher LIMIT {limit} OFFSET {offset}"
            df_temp = pd.read_sql(sql, engine)
            rows = df_temp.shape[0]
            df_temp.to_parquet(data_dir / f"trencher_raw_{offset}_{offset + limit}.parquet")
            offset += limit
            logging.info(f"Progress: {round(offset / 10074821, 2) * 100}%")
            if rows == 0:
                break
