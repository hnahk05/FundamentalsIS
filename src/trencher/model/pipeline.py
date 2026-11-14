import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from ..config import load_config, resolve_path

sns.set()
pd.options.mode.chained_assignment = None


class Pipeline:
    def __init__(
        self,
        X_agg: list[dict],
        y_agg: dict,
        model_config: dict,
        cables: pd.DataFrame,
        kp_name: str = "NAV_DATA_KP",
        data_path: str | Path = "data_constructed",
        trim_observations: int = 200,
    ):
        self.train_data = pd.DataFrame()
        self.cables = cables["Cycle"].tolist()
        self.kp_name = kp_name
        self.data_path = Path(data_path)
        self.trim_observations = trim_observations

        self.current_cable_name = ""
        self.current_cable_raw = pd.DataFrame()
        self.current_cable_aggregated = pd.DataFrame()
        self.current_cable_train_data = pd.DataFrame()

        self.X_agg = X_agg
        self.y_agg = y_agg
        self.model_config = model_config

        self.RMSE_list: list[float] = []
        self.df_evaluation = pd.DataFrame()
        self.coef_df = pd.DataFrame()
        self.variances: list[float] = []

    def read_cable(self, cable: str) -> None:
        self.current_cable_name = cable
        file_path = self.data_path / f"EXPORT {cable}.csv"
        self.current_cable_raw = pd.read_csv(file_path).iloc[
            self.trim_observations : -self.trim_observations
        ]
        print(f"Cable Read: {self.current_cable_raw.shape} observations")
        self.current_cable_raw = self.current_cable_raw.set_index(self.kp_name)

    def aggregate_current_cable(self) -> None:
        df_agg = pd.DataFrame()

        for agg_dict in self.X_agg:
            varname = f"{agg_dict['varname']}_{agg_dict['aggregation']}" if agg_dict.get('aggregation') else agg_dict['varname']

            if agg_dict.get("standardize"):
                series_temp = (
                    self.current_cable_raw[agg_dict["varname"]]
                    - self.current_cable_raw[agg_dict["varname"]].mean()
                ) / self.current_cable_raw[agg_dict["varname"]].std()
            elif agg_dict.get("aggregation") == "":
                series_temp = self.current_cable_raw[agg_dict["varname"]]
            elif agg_dict["aggregation"] == "diff":
                series_temp = self.current_cable_raw[agg_dict["varname"]].diff()
            else:
                series_temp = self.current_cable_raw[agg_dict["varname"]].shift(
                    agg_dict["window"][0]
                )
                if agg_dict["aggregation"] == "mean":
                    series_temp = series_temp.rolling(window=agg_dict["window"][1]).mean()
                elif agg_dict["aggregation"] == "var":
                    series_temp = series_temp.rolling(window=agg_dict["window"][1]).var()
                elif agg_dict["aggregation"] == "avgdiff":
                    series_temp = series_temp.diff()
                    series_temp = series_temp.rolling(window=agg_dict["window"][1]).mean()

            series_temp.rename(varname, inplace=True)
            df_agg = df_agg.merge(
                right=series_temp, how="outer", left_index=True, right_index=True
            )

        series_temp = self.current_cable_raw[self.y_agg["varname"]]
        if self.y_agg.get("standardize"):
            series_temp = (series_temp - series_temp.mean()) / series_temp.std()

        if self.y_agg["aggregation"] == "diff":
            series_temp = series_temp.diff()
        elif self.y_agg["aggregation"] != "":
            series_temp = series_temp.shift(self.y_agg["window"][0])
            if self.y_agg["aggregation"] == "mean":
                series_temp = series_temp.rolling(window=self.y_agg["window"][1]).mean()
            elif self.y_agg["aggregation"] == "var":
                series_temp = series_temp.rolling(window=self.y_agg["window"][1]).var()
            elif self.y_agg["aggregation"] == "max":
                series_temp = series_temp.rolling(window=self.y_agg["window"][1]).max()

        df_agg = df_agg.merge(
            right=series_temp.rename(self.y_agg["varname"]),
            how="outer",
            left_index=True,
            right_index=True,
        )
        self.current_cable_aggregated = df_agg.dropna()

    def train_linear_model(self, logistic: bool = True, on_full_dataset: bool = True) -> None:
        if on_full_dataset:
            train_data = self.train_data.copy()
            print(f"{self.current_cable_name} Training on {train_data.shape} observations.")
        else:
            train_data = self.current_cable_train_data.copy()
            print(f"{self.current_cable_name} Training on {train_data.shape} observations.")

        X = train_data.loc[:, ~train_data.columns.isin([self.y_agg["varname"]])]
        y = train_data[self.y_agg["varname"]]

        poly = PolynomialFeatures(
            interaction_only=self.model_config["interaction_only"],
            include_bias=False,
            degree=self.model_config["degree"],
        )
        X_poly = poly.fit_transform(X)

        if logistic:
            try:
                if self.model_config.get("regularization"):
                    model = LogisticRegression(
                        fit_intercept=True,
                        penalty=self.model_config["regularization"],
                        C=self.model_config["C"],
                        solver="liblinear",
                    )
                else:
                    model = LogisticRegression(fit_intercept=True, solver="liblinear")
            except ValueError:
                print(f"{self.current_cable_name} not 2 classes in cable.")
                return
        else:
            model = Lasso(
                alpha=self.model_config.get("alpha", 1.0),
                max_iter=self.model_config.get("max_iter", 1000),
                tol=0.001,
            )

        model.fit(X_poly, y)
        self.poly = poly
        if on_full_dataset:
            print(f"On full dataset: {model.coef_}")
            self.model = model
        else:
            print(f"On partial dataset: {model.coef_}")
            self.partial_model = model

    def add_current_cable_train_data(self, folds: list[int] | None = None) -> None:
        if folds is None:
            folds = [1, 2, 3, 4]
        df_temp = self.current_cable_aggregated.copy()
        total_rows = len(df_temp)
        fold_size = total_rows // self.model_config["folds"]

        for fold in folds:
            start_idx = (fold - 1) * fold_size
            end_idx = fold * fold_size
            df_fold = df_temp.iloc[start_idx:end_idx].copy()
            df_fold["Cable"] = self.current_cable_name
            df_fold = df_fold.set_index(["Cable"], append=True)
            self.train_data = pd.concat([self.train_data, df_fold])
            self.current_cable_train_data = pd.concat(
                [self.current_cable_train_data, df_fold]
            )

    def predict_linear_model(self, on_train: bool = True, test_fold: int | None = None) -> None:
        if on_train:
            test_data = self.train_data
        else:
            total_rows = len(self.current_cable_aggregated)
            fold_size = total_rows // self.model_config["folds"]
            start_idx = (test_fold - 1) * fold_size
            end_idx = test_fold * fold_size
            test_data = self.current_cable_aggregated.iloc[start_idx:end_idx].copy()

        cable = self.current_cable_name
        X = test_data.loc[:, ~test_data.columns.isin([self.y_agg["varname"]])]
        y = test_data.loc[:, self.y_agg["varname"]]
        print(f"Testing on {test_data.shape} observations")

        X_poly = self.poly.transform(X)
        y_pred = self.model.predict(X_poly)

        result_df = pd.DataFrame(
            {
                "cable": cable,
                self.kp_name: test_data.index.get_level_values(self.kp_name),
                "Actual": y,
                "Predicted": y_pred,
            }
        )

        RMSE = math.sqrt(np.square(np.subtract(result_df["Actual"], result_df["Predicted"])).mean())
        self.RMSE_list.append(RMSE)
        self.df_evaluation = pd.concat([self.df_evaluation, result_df])

    def plot_results(self) -> None:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(self.df_evaluation[self.kp_name], self.df_evaluation["Actual"], label="Actual")
        ax.plot(
            self.df_evaluation[self.kp_name], self.df_evaluation["Predicted"], label="Predicted"
        )
        if (
            self.df_evaluation[self.kp_name].iloc[0]
            > self.df_evaluation[self.kp_name].iloc[-1]
        ):
            ax.invert_xaxis()

        for i in range(0, 5):
            ax.axvline(
                x=self.df_evaluation[self.kp_name].iloc[0]
                + i
                * (
                    self.df_evaluation[self.kp_name].iloc[-1]
                    - self.df_evaluation[self.kp_name].iloc[0]
                )
                / 4,
                color="grey",
                linestyle="--",
            )

        ax.set_xlabel(self.kp_name)
        ax.set_ylabel(self.y_agg["varname"])
        ax.set_title(f"{self.y_agg['varname']} vs {self.kp_name}")
        ax.legend()
        plt.show()

    def return_coefficients(self, on_full_dataset: bool = True) -> None:
        x_columnlist = [
            col for col in self.train_data.columns if col != self.y_agg["varname"]
        ]

        if on_full_dataset:
            coef_df_temp = pd.DataFrame(
                {
                    "name": self.poly.get_feature_names_out(x_columnlist),
                    "train_data": "Overall",
                    "cable": self.current_cable_name,
                    "observations": self.train_data.shape[0],
                    "coef": self.model.coef_.ravel(),
                    "intercept": self.model.intercept_[0],
                    "var": np.nan,
                }
            )
        else:
            coef_df_temp = pd.DataFrame(
                {
                    "name": self.poly.get_feature_names_out(x_columnlist),
                    "train_data": "Cable-specific",
                    "cable": self.current_cable_name,
                    "observations": self.current_cable_train_data.shape[0],
                    "coef": self.partial_model.coef_.ravel(),
                    "intercept": self.partial_model.intercept_[0],
                    "var": np.nan,
                }
            )

        coef_df_temp.set_index("name", inplace=True)
        self.coef_df = pd.concat([self.coef_df, coef_df_temp])

    def loop_cables(self, logistic: bool = True, cablecount: int = 3) -> None:
        cable_counter = 0
        for cable in tqdm(self.cables):
            cable_counter += 1
            if cable_counter > cablecount:
                break
            print(cable)
            self.read_cable(cable)
            self.aggregate_current_cable()
            self.add_current_cable_train_data()

            self.train_linear_model(logistic, on_full_dataset=True)
            self.return_coefficients(on_full_dataset=True)
            try:
                self.train_linear_model(logistic, on_full_dataset=False)
                self.return_coefficients(on_full_dataset=False)
            except ValueError as e:
                print(e)
                print(f"{cable}: no positive classes.")
                continue

            self.model = None
            self.partial_model = None
            self.current_cable_train_data = pd.DataFrame()

        self.cable_variance_df = (
            self.train_data.reset_index().groupby("Cable").var().reset_index()
        )

    def plot_coefficients(self, coef_name: str) -> None:
        print(coef_name)
        coefficients = self.coef_df.loc[coef_name, :]

        fig, ax1 = plt.subplots(figsize=(14, 7))

        coefficients_cablespecific = coefficients[coefficients["train_data"] == "Cable-specific"]
        coefficients_overall = coefficients[coefficients["train_data"] == "Overall"]
        coefficients_plot = pd.merge(
            left=coefficients_cablespecific,
            right=coefficients_overall,
            how="left",
            left_on="cable",
            right_on="cable",
        )

        ax1.plot(
            coefficients_plot["cable"],
            coefficients_plot["coef_x"],
            marker="o",
            linestyle="-",
            label="Cable Coefficient",
            color="lightslategray",
        )
        ax1.plot(
            coefficients_plot["cable"],
            coefficients_plot["coef_y"],
            marker="o",
            linestyle="-",
            label="Overall Coefficient",
            color="darkblue",
        )

        coefficients_plot = pd.merge(
            left=coefficients_plot,
            right=self.cable_variance_df,
            left_on="cable",
            right_on="Cable",
            how="left",
        )

        ax1.set_xlabel("Cable")
        ax1.set_ylabel("Coefficient")
        ax1.legend(loc="upper left")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax1.tick_params(axis="x", rotation=90)

        if " " not in coef_name:
            ax2 = ax1.twinx()
            ax2.plot(
                coefficients_plot["Cable"],
                coefficients_plot[coef_name],
                color="green",
                marker="o",
                linestyle="-",
                label="Variance",
            )
            ax2.set_ylabel("Variance")
            ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.show()


def create_pipeline_from_config(config_path: str | Path | None = None) -> Pipeline:
    cfg = load_config(config_path)
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    data_dir = resolve_path(base, cfg["paths"]["data_constructed"])

    intervals_csv = data_dir.parent / cfg["paths"]["data"] / "southfork_intervals_inc.csv"
    cables = pd.read_csv(intervals_csv)
    cables["Start UTC"] = pd.to_datetime(cables["Start UTC"])
    cables = cables.sort_values(by="Start UTC").reset_index()

    m = cfg["model"]
    return Pipeline(
        X_agg=m["X_agg"],
        y_agg=m["y_agg"],
        model_config=m["model_config"],
        cables=cables,
        kp_name=cfg["kp_name"],
        data_path=data_dir,
        trim_observations=m.get("trim_observations", 200),
    )
