from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from algaie.training.foundation_train import train_foundation_model
from backend.scripts._cli_utils import load_pipeline_config, prepare_run, write_artifact_log


def run(config_path: str, input_paths: list[Path]) -> None:
    config = load_pipeline_config(config_path)
    paths, registry, run_dir = prepare_run(config)
    canonical = pd.concat([pd.read_parquet(path) for path in input_paths], ignore_index=True)
    destination = paths.models_foundation / "foundation_model.txt"
    train_foundation_model(canonical, destination)
    registry.register("foundation_model", destination, "v1")
    write_artifact_log(registry, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path.strip()) for path in args.inputs.split(",")]
    run(args.config, input_paths)


if __name__ == "__main__":
    main()
