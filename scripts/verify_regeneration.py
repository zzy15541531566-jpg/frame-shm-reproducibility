"""Regenerate all rows in memory; never overwrite the supplied CSV."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import DATA_PATH, LABEL_COLUMNS, FEATURE_COLUMNS
from src.generate_dataset import generate

def main():
    original = pd.read_csv(DATA_PATH)
    regenerated = generate()
    if original.shape != regenerated.shape or list(original.columns) != list(regenerated.columns):
        raise SystemExit("FAIL: dataset schema mismatch")
    if not original[LABEL_COLUMNS].equals(regenerated[LABEL_COLUMNS]):
        raise SystemExit("FAIL: label values or row order mismatch")
    a = original[FEATURE_COLUMNS].to_numpy()
    b = regenerated[FEATURE_COLUMNS].to_numpy()
    error = float(np.max(np.abs(a-b)))
    print("Maximum absolute feature difference: %.12g" % error)
    if not np.allclose(a, b, rtol=1e-9, atol=1e-12):
        raise SystemExit("FAIL: features differ beyond tolerance")
    print("PASS: all 1,560 rows match within the comparison tolerance. Original CSV unchanged.")

if __name__ == "__main__":
    main()
