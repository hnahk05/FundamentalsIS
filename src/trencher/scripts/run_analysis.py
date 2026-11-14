#!/usr/bin/env python
import argparse

from trencher.analysis.descriptive import plot_descriptive_analysis
from trencher.analysis.correlation import correlation_analysis_all
from trencher.analysis.rq2 import plot_rq2_boxplot
from trencher.analysis.southfork import plot_southfork_soil_layers


def main():
    parser = argparse.ArgumentParser(description="Run Trencher analysis scripts")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--analysis",
        choices=["descriptive", "correlation", "rq2", "southfork", "all"],
        default="all",
        help="Which analysis to run",
    )
    args = parser.parse_args()

    if args.analysis in ("descriptive", "all"):
        print("Running descriptive analysis...")
        plot_descriptive_analysis(args.config)

    if args.analysis in ("correlation", "all"):
        print("Running correlation analysis...")
        correlation_analysis_all(config_path=args.config)

    if args.analysis in ("rq2", "all"):
        print("Running RQ2 analysis...")
        plot_rq2_boxplot(config_path=args.config)

    if args.analysis in ("southfork", "all"):
        print("Running South Fork soil layer analysis...")
        plot_southfork_soil_layers(args.config)


if __name__ == "__main__":
    main()
