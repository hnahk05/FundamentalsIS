#!/usr/bin/env python
import argparse

from trencher.config import load_config
from trencher.data.cleaner import clean_all_cables
from trencher.data.constructor import construct_all_cables
from trencher.data.aggregator import aggregate_all_cables
from trencher.model.pipeline import create_pipeline_from_config


def main():
    parser = argparse.ArgumentParser(description="Run the Trencher ML pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--cable-count", type=int, default=100, help="Number of cables to process")
    parser.add_argument("--logistic", action="store_true", default=True, help="Use logistic regression")
    parser.add_argument("--skip-clean", action="store_true", help="Skip data cleaning step")
    parser.add_argument("--skip-construct", action="store_true", help="Skip data construction step")
    parser.add_argument("--skip-aggregate", action="store_true", help="Skip data aggregation step")
    args = parser.parse_args()

    if not args.skip_clean:
        print("Cleaning cable sensor data...")
        clean_all_cables(args.config)

    if not args.skip_construct:
        print("Constructing cable features...")
        construct_all_cables(args.config)

    if not args.skip_aggregate:
        print("Aggregating cable data...")
        aggregate_all_cables(args.config)

    print("Running ML pipeline...")
    pipeline = create_pipeline_from_config(args.config)
    pipeline.loop_cables(logistic=args.logistic, cablecount=args.cable_count)
    pipeline.coef_df.to_csv("coefficients.csv", sep=";", decimal=",")

    cfg = load_config(args.config)
    from pathlib import Path
    base = Path(args.config).resolve().parent if args.config else Path.cwd()
    from trencher.config import resolve_path
    desc_dir = resolve_path(base, cfg["paths"]["descriptive_analysis"])
    desc_dir.mkdir(parents=True, exist_ok=True)
    pipeline.train_data.to_csv(desc_dir / "southfork.csv")

    print("Pipeline complete. Coefficients saved to coefficients.csv")


if __name__ == "__main__":
    main()
