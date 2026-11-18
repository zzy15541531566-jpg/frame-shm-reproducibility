from __future__ import annotations

import argparse

from .analyze_features import run as run_feature_analysis
from .check_dataset import evaluate_checks
from .common import load_dataset, mode_dirs
from .generate_dataset import main as generate_dataset
from .plot_outputs import run as run_plots
from .run_validation import run as run_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    generate_dataset()
    data = load_dataset()
    _, _, check_dir = mode_dirs(arguments.quick)
    checks = evaluate_checks(data)
    import pandas as pd

    pd.DataFrame(checks).to_csv(
        check_dir / "dataset_checks.csv", index=False, encoding="utf-8-sig"
    )
    if not all(item["passed"] for item in checks):
        raise SystemExit(1)
    run_validation(arguments.quick)
    run_feature_analysis(arguments.quick)
    run_plots(arguments.quick)


if __name__ == "__main__":
    main()
