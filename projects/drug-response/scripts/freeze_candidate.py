#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_sciplex import file_sha256, project_path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selection-evidence", nargs="+", type=Path, required=True)
    parser.add_argument("--selection-rule", required=True)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = {
        "dataset": project_path(config["dataset_path"]),
        "compound": project_path(config["compound_path"]),
        "gene_panel": project_path(config["gene_panel_path"]),
        "split": project_path(config["split_path"]),
    }
    evidence_paths = [path if path.is_absolute() else ROOT / path for path in args.selection_evidence]
    missing = [str(path) for path in [config_path, *inputs.values(), *evidence_paths] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze candidate; missing files: {missing}")

    manifest = {
        "schema_version": 1,
        "status": "frozen_before_test",
        "candidate": args.candidate,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": args.selection_rule,
        "config": {
            "path": relative_path(config_path),
            "sha256": file_sha256(config_path),
        },
        "inputs": {
            name: {"path": relative_path(path), "sha256": file_sha256(path)}
            for name, path in inputs.items()
        },
        "selection_evidence": [
            {"path": relative_path(path), "sha256": file_sha256(path)}
            for path in evidence_paths
        ],
        "ensemble_seeds": args.seeds,
        "test_evaluations_before_freeze": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
