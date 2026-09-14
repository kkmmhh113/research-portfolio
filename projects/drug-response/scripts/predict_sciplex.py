#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_star.chemistry import molecular_descriptors, morgan_fingerprint, standardize_smiles
from my_star.data import ContextPrior, TargetScaler
from my_star.model import StructureConditionedPerturbationModel, model_kwargs_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        help="Repeat for seed-ensemble inference",
    )
    checkpoint_group.add_argument(
        "--ensemble-summary",
        type=Path,
        help="Seed-sweep summary containing checkpoint paths and ensemble calibration",
    )
    parser.add_argument(
        "--ensemble-residual-scale",
        type=float,
        default=1.0,
        help="Validation-fitted scale applied after averaging checkpoint predictions",
    )
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--dose-nm", type=float, required=True)
    parser.add_argument("--duration-hours", type=float, required=True)
    parser.add_argument("--cell-line", required=True)
    parser.add_argument(
        "--baseline-npy",
        type=Path,
        required=True,
        help="Matched untreated baseline expression in checkpoint gene-panel order",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.ensemble_summary is not None:
        summary_path = (
            args.ensemble_summary
            if args.ensemble_summary.is_absolute()
            else ROOT / args.ensemble_summary
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint_paths = [
            ROOT / Path(str(run["checkpoint"]).replace("\\", "/"))
            for run in summary["individual_runs"]
        ]
        ensemble_residual_scale = float(summary["ensemble"]["residual_scale"])
    else:
        assert args.checkpoint is not None
        checkpoint_paths = [path if path.is_absolute() else ROOT / path for path in args.checkpoint]
        ensemble_residual_scale = float(args.ensemble_residual_scale)

    structure = standardize_smiles(args.smiles)
    if structure is None:
        raise SystemExit("Invalid SMILES")
    baseline = np.load(args.baseline_npy).astype(np.float32)
    context = torch.tensor(
        [
            [
                (float(np.log10(args.dose_nm * 1e-9)) + 6.5) / 1.5,
                np.log2(1.0 + args.duration_hours) / 7.0,
            ]
        ],
        dtype=torch.float32,
    )
    predictions = []
    reference_checkpoint = None
    context_prior = None
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if reference_checkpoint is None:
            reference_checkpoint = checkpoint
        elif (
            checkpoint["gene_panel"] != reference_checkpoint["gene_panel"]
            or checkpoint["cell_map"] != reference_checkpoint["cell_map"]
        ):
            raise SystemExit("Ensemble checkpoints use incompatible gene panels or cell maps")
        cell_map = checkpoint["cell_map"]
        if args.cell_line not in cell_map:
            raise SystemExit(
                f"Unknown phase-1 cell line: {args.cell_line}; available={sorted(cell_map)}"
            )
        output_dim = int(checkpoint["output_dim"])
        if baseline.shape != (output_dim,):
            raise SystemExit(f"Expected baseline shape {(output_dim,)}, got {baseline.shape}")
        config = checkpoint["config"]
        descriptors = torch.from_numpy(
            molecular_descriptors(
                structure.canonical_smiles,
                str(config.get("descriptor_set", "basic")),
            )
        ).unsqueeze(0)
        fingerprint = torch.from_numpy(
            morgan_fingerprint(
                structure.canonical_smiles,
                radius=int(config.get("fingerprint_radius", 2)),
                use_counts=bool(config.get("fingerprint_use_counts", False)),
                include_chirality=bool(config.get("fingerprint_include_chirality", False)),
            )
        ).unsqueeze(0)
        model = StructureConditionedPerturbationModel(
            output_dim=output_dim,
            num_cell_lines=len(cell_map),
            response_basis=checkpoint.get("response_basis"),
            **model_kwargs_from_config(config),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            scaled_residual = model(
                fingerprint,
                descriptors,
                torch.tensor([cell_map[args.cell_line]], dtype=torch.long),
                context,
                torch.from_numpy(baseline).unsqueeze(0),
            ).numpy()[0]
        scaler = TargetScaler.from_dict(checkpoint["target_scaler"])
        prior = ContextPrior.from_dict(checkpoint["context_prior"])
        current_prior = prior.lookup(args.cell_line, args.dose_nm, args.duration_hours)
        if context_prior is None:
            context_prior = current_prior
        elif not np.allclose(context_prior, current_prior):
            raise SystemExit("Ensemble checkpoints use incompatible context priors")
        residual = scaler.inverse_transform(scaled_residual)
        predictions.append(
            current_prior + float(checkpoint.get("residual_scale", 1.0)) * residual
        )
    assert reference_checkpoint is not None and context_prior is not None
    mean_prediction = np.mean(predictions, axis=0)
    predicted = context_prior + ensemble_residual_scale * (mean_prediction - context_prior)
    output = pd.DataFrame(
        {
            "ensembl_id": reference_checkpoint["gene_panel"],
            "predicted_delta_y": predicted.astype(np.float32),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        output.to_parquet(args.output, index=False)
    elif args.output.suffix.lower() == ".csv":
        output.to_csv(args.output, index=False)
    else:
        raise SystemExit("Output must end in .csv or .parquet")
    print(
        f"Saved {len(output)} predicted gene responses from {len(checkpoint_paths)} "
        f"checkpoint(s) -> {args.output}"
    )


if __name__ == "__main__":
    main()
