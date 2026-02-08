from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from algaie.core.config import PipelineConfig
from algaie.core.paths import ArtifactPaths
from algaie.data.canonical.validate import quarantine_invalid_returns, validate_canonical_daily
from algaie.data.canonical.writer import write_canonical_daily
from backend.scripts._cli_utils import load_pipeline_config, prepare_run, write_artifact_log


def run(config: PipelineConfig, input_path: Path, ticker: str) -> None:
    paths, registry, run_dir = prepare_run(config)
    df = pd.read_parquet(input_path)
    issues = validate_canonical_daily(df)
    if issues:
        raise RuntimeError("; ".join(issue.message for issue in issues))
    report_path = paths.reports / "data_quality" / "invalid_returns.parquet"
    valid_df, _ = quarantine_invalid_returns(df, config, report_path)
    destination = paths.canonical_daily / f"ticker={ticker}" / "data.parquet"
    write_canonical_daily(valid_df, destination)
    registry.register("canonical_daily", destination, "v1")
    write_artifact_log(registry, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--ticker", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    run(config, Path(args.input), args.ticker)


if __name__ == "__main__":
    main()
