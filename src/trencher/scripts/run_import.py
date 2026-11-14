#!/usr/bin/env python
import argparse

from trencher.config import load_config
from trencher.data.importer import import_sensor_data, import_postgres_tables


def main():
    parser = argparse.ArgumentParser(description="Import data from OSIsoft PI and PostgreSQL")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--source", choices=["pi", "postgres", "all"], default="all", help="Data source to import from")
    args = parser.parse_args()

    if args.source in ("pi", "all"):
        print("Importing sensor data from OSIsoft PI...")
        import_sensor_data(args.config)

    if args.source in ("postgres", "all"):
        print("Importing data from PostgreSQL...")
        import_postgres_tables(args.config)


if __name__ == "__main__":
    main()
