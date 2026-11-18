"""Verify the archived data, checksums and main result summaries without training."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_dataset
from src.check_dataset import evaluate_checks

def main():
    data = load_dataset()
    checks = evaluate_checks(data)
    failed = [item for item in checks if not item["passed"]]
    if failed:
        raise SystemExit("Dataset checks failed: " + str(failed))
    manifest = json.loads((ROOT / "docs/archive_sha256.json").read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit("Archive checksum mismatch: " + name)
    table = ROOT / "reference_results/tables"
    detail = pd.read_csv(table / "baseline_detail.csv")
    summary = pd.read_csv(table / "baseline_summary.csv")
    for _, row in summary.iterrows():
        values = detail.loc[detail.model == row.model, "test_accuracy"]
        if len(values) != 10 or not np.isclose(values.mean(), row.test_accuracy_mean):
            raise SystemExit("Baseline summary mismatch: " + row.model)
    matrix = pd.read_csv(table / "pooled_confusion_matrix.csv", index_col=0).to_numpy()
    main = pd.read_csv(table / "split_metrics.csv")
    if matrix.sum() != 3120 or not np.isclose(np.trace(matrix) / matrix.sum(), main.test_accuracy.mean()):
        raise SystemExit("Confusion matrix mismatch")
    print("PASS: dataset checks; archived checksums; ten-run baseline summaries; pooled accuracy.")
    print("Rows: %d. Features: 28. This is an archive check, not a model rerun." % len(data))

if __name__ == "__main__":
    main()
