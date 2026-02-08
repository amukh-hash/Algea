from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from algaie.training.ranker_train import train_ranker_model
from backend.scripts._cli_utils import load_pipeline_config, prepare_run, write_artifact_log


def run(config_path: str, features_path: Path) -> None:
    config = load_pipeline_config(config_path)
    paths, registry, run_dir = prepare_run(config)
    features = pd.read_parquet(features_path)
    destination = paths.models_ranker / "ranker_model.txt"
    train_ranker_model(features, destination)
    registry.register("ranker_model", destination, "v1")
    write_artifact_log(registry, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, Path(args.features))


if __name__ == "__main__":
    main()
