#!/usr/bin/env python3
"""Run participant-bootstrap within/between significance tests from a CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bootstrap_criteria import participant_bootstrap_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Segment-level CSV file.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--participant-col", default="pid")
    parser.add_argument("--score-col", default="score")
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        required=True,
        help="Feature columns to test, e.g. AU01_r AU02_r AU04_r.",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    results = participant_bootstrap_inference(
        df,
        args.feature_cols,
        participant_col=args.participant_col,
        score_col=args.score_col,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
