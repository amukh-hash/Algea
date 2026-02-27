from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from algea.data.priors.build import build_priors, write_priors
from algea.models.foundation.chronos2 import FoundationModelConfig
from backend.scripts._cli_utils import load_pipeline_config, prepare_run, write_artifact_log


def run(config_path: str, input_paths: list[Path], asof: date) -> None:
    config = load_pipeline_config(config_path)
    paths, registry, run_dir = prepare_run(config)
    canonical = pd.concat([pd.read_parquet(path) for path in input_paths], ignore_index=True)
    priors = build_priors(canonical, asof, FoundationModelConfig(enable_quantiles=config.enable_quantiles))
    destination = paths.priors / f"date={asof}" / "priors.parquet"
    write_priors(priors, destination)
    registry.register("priors", destination, "v1")
    write_artifact_log(registry, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--asof", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path.strip()) for path in args.inputs.split(",")]
    run(args.config, input_paths, date.fromisoformat(args.asof))


if __name__ == "__main__":
    main()
