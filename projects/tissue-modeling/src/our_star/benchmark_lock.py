"""Hash verification and prespecified acceptance gates for locked benchmarks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "locked_input_sha256" not in payload:
        raise ValueError("invalid benchmark lock manifest")
    return payload


def verify_locked_inputs(
    lock: Mapping[str, Any], project_root: str | Path
) -> dict[str, str]:
    """Fail closed when any code/config/data-development input changed."""

    root = Path(project_root)
    verified: dict[str, str] = {}
    errors: list[str] = []
    for relative, expected in lock["locked_input_sha256"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        actual = sha256_file(path)
        verified[relative] = actual
        if actual != expected:
            errors.append(f"hash mismatch: {relative}")
    if errors:
        raise RuntimeError("locked benchmark inputs changed; " + "; ".join(errors))
    return verified


def evaluate_acceptance_gates(
    metrics: Mapping[str, Any], criteria: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Evaluate metric rules without changing or interpreting the criteria."""

    results: dict[str, dict[str, Any]] = {}
    for metric, rule in criteria.items():
        if metric not in metrics:
            raise KeyError(f"acceptance metric is missing: {metric}")
        value = float(metrics[metric])
        operator = rule["operator"]
        if not np.isfinite(value):
            passed = False
        elif operator == "maximum":
            passed = value <= float(rule["value"])
        elif operator == "minimum":
            passed = value >= float(rule["value"])
        elif operator == "inclusive_range":
            passed = float(rule["lower"]) <= value <= float(rule["upper"])
        else:
            raise ValueError(f"unsupported acceptance operator: {operator}")
        results[metric] = {
            "value": value,
            "rule": dict(rule),
            "passed": bool(passed),
        }
    return results, all(result["passed"] for result in results.values())
