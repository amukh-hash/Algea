from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from algaie.data.eligibility.build import build_eligibility, load_canonical_files, write_eligibility
from backend.scripts._cli_utils import load_pipeline_config, prepare_run, write_artifact_log


def run(input_paths: list[Path], asof: date, config_path: str) -> None:
    config = load_pipeline_config(config_path)
    paths, registry, run_dir = prepare_run(config)
    canonical = load_canonical_files(input_paths)
    frame = build_eligibility(canonical, asof)
    destination = paths.eligibility / f"asof={asof}" / "eligibility.parquet"
    write_eligibility(frame, destination)
    registry.register("eligibility", destination, "v1")
    write_artifact_log(registry, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--inputs", required=True, help="Comma-separated canonical parquet paths")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path.strip()) for path in args.inputs.split(",")]
    run(paths, date.fromisoformat(args.asof), args.config)


if __name__ == "__main__":
    main()
