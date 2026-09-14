#!/usr/bin/env python3
"""Replay synthetic metric checks only; no dataset, training or test-access mode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import bootstrap, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from my_star.paper_a_metrics import PearsonPolicy, Predictions, Reference, evaluate_paired, compound_macro_bootstrap

OUTPUT = ROOT / "artifacts/paper_a_metrics_synthetic_audit_20260906.json"
CODE = ("scripts/audit_paper_a_metrics.py", "src/my_star/paper_a_metrics.py", "tests/test_paper_a_metrics.py")


def encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def audit() -> dict:
    a = np.array([1., -1., 0.]) / np.sqrt(2.)
    b = np.array([1., 1., -2.]) / np.sqrt(6.)
    gains = [.1] * 9 + [-.5]
    metadata = pd.DataFrame([
        {"sample_id": f"synthetic-{i:02}", "compound_id": "synthetic-a" if i < 9 else "synthetic-b",
         "fold_id": 1 if i < 9 else 2, "cell_line": "SYNTHETIC", "dose_nM": float(i+1), "duration_hours": 24.}
        for i in range(10)])
    genes = ("synthetic-g1", "synthetic-g2", "synthetic-g3")
    target = np.tile(a, (10, 1))
    candidate = np.stack([r*a + np.sqrt(1-r*r)*b for r in gains])
    baseline = np.tile(b, (10, 1))
    reference = Reference(metadata, genes, target, np.ones(target.shape, dtype=bool))
    report, rows, compounds = evaluate_paired(
        reference, Predictions(metadata.copy(), genes, candidate), Predictions(metadata.copy(), genes, baseline),
        allowed_folds=(1, 2), pearson_policy=PearsonPolicy("zero_with_flag", 1e-12))
    np.testing.assert_allclose(report["primary"]["observed_mean_difference"], -.2, rtol=0, atol=1e-14)
    np.testing.assert_allclose(report["secondary_row_micro_paired_gain"], .04, rtol=0, atol=1e-14)
    np.testing.assert_allclose(rows.candidate_pearson_score, pearsonr(target, candidate, axis=1).statistic,
                               rtol=0, atol=1e-14)
    effects = {"a": -.5, "b": .8, "c": .1, "d": .3}
    independent = bootstrap((np.array(list(effects.values())),), np.mean, method="percentile",
                            n_resamples=10_000, rng=np.random.Generator(np.random.PCG64(2026)))
    calculated = compound_macro_bootstrap(effects)
    np.testing.assert_array_equal([calculated["ci95_lower"], calculated["ci95_upper"]],
                                  independent.confidence_interval)
    return {
        "schema_version": 1, "status": "passed_synthetic_metric_checks_not_scientific_results",
        "fixture": {"source": "constructed_unit_vectors_not_measured_biology", "conditions": 10,
                    "compounds": 2, "genes": 3, "condition_counts_per_compound": [9, 1],
                    "intended_compound_gains": [.1, -.5], "expected_compound_macro": -.2,
                    "expected_row_micro": .04},
        "calculated": {"compound_macro": report["primary"]["observed_mean_difference"],
                       "row_micro": report["secondary_row_micro_paired_gain"],
                       "compound_gains": compounds.paired_gain.tolist()},
        "independent_scipy_sample_Pearson_check": "passed_atol_1e-14",
        "independent_scipy_percentile_check": "passed_exact_endpoints",
        "bootstrap_settings": {"resamples": 10_000, "seed": 2026, "rng": "PCG64"},
        "dataset_or_prediction_files_read": False, "real_scientific_result_computed": False,
        "training_performed": False, "lincs_accessed": False, "protocol_approval": "pending",
        "limits": ["Supplied-array numerical kernel, not an authorized or frozen production evaluator.",
                   "Constant/low-denominator handling is explicit but remains a protocol decision.",
                   "Bootstrap is conditional on fixed predictions, not training/seed/scaffold resampling.",
                   "Positive bootstrap fraction is neither a p-value nor posterior probability.",
                   "No real v1.4 population, gene axis, mask or prediction provenance is established here.",
                   "Calibration, secondary error/top-k metrics, promotion gates and one-time access are not implemented here."],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__},
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and OUTPUT.exists():
        raise FileExistsError(f"Preserve existing synthetic audit: {OUTPUT}")
    report = audit()
    data = encoded(report)
    if args.verify:
        if OUTPUT.read_bytes() != data:
            raise ValueError("Synthetic audit does not exactly replay")
    else:
        with OUTPUT.open("xb") as handle:
            handle.write(data)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
