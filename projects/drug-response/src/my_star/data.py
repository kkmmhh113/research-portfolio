from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .chemistry import molecular_descriptors, morgan_fingerprint
from .splits import validate_split_manifest


@dataclass
class TargetScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, targets: np.ndarray, mode: str = "per_gene") -> "TargetScaler":
        mean = targets.mean(axis=0).astype(np.float32)
        if mode == "per_gene":
            scale = targets.std(axis=0).astype(np.float32)
        elif mode == "global":
            scale = np.full(targets.shape[1], targets.std(), dtype=np.float32)
        else:
            raise ValueError(f"Unknown target scaling mode: {mode}")
        scale = np.maximum(scale, 0.05)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "TargetScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
        )


@dataclass
class ContextPrior:
    values: dict[tuple[str, float, float], np.ndarray]
    fallback: np.ndarray

    @classmethod
    def fit(cls, observations: pd.DataFrame) -> "ContextPrior":
        values = {
            (str(cell), float(dose), float(duration)): np.stack(group["delta_y"].map(np.asarray))
            .mean(axis=0)
            .astype(np.float32)
            for (cell, dose, duration), group in observations.groupby(
                ["cell_line", "dose_nM", "duration_hours"]
            )
        }
        fallback = np.stack(observations["delta_y"].map(np.asarray)).mean(axis=0).astype(np.float32)
        return cls(values=values, fallback=fallback)

    def lookup(self, cell_line: str, dose_nM: float, duration_hours: float) -> np.ndarray:
        return self.values.get(
            (str(cell_line), float(dose_nM), float(duration_hours)),
            self.fallback,
        )

    def to_dict(self) -> dict:
        return {
            "values": {
                json.dumps(key): value.tolist()
                for key, value in self.values.items()
            },
            "fallback": self.fallback.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ContextPrior":
        values = {
            tuple(json.loads(key)): np.asarray(value, dtype=np.float32)
            for key, value in payload["values"].items()
        }
        normalized = {
            (str(cell), float(dose), float(duration)): value
            for (cell, dose, duration), value in values.items()
        }
        return cls(normalized, np.asarray(payload["fallback"], dtype=np.float32))


def load_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "sample_id",
        "compound_id",
        "canonical_smiles",
        "log10_dose_molar",
        "duration_hours",
        "cell_line",
        "baseline_expression",
        "delta_y",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    return frame


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class SciPlexDataset(Dataset):
    def __init__(
        self,
        observations: pd.DataFrame,
        compound_ids: set[str],
        cell_map: dict[str, int],
        target_scaler: TargetScaler,
        context_prior: ContextPrior,
        fingerprint_radius: int = 2,
        fingerprint_use_counts: bool = False,
        fingerprint_include_chirality: bool = False,
        descriptor_set: str = "basic",
    ) -> None:
        self.frame = observations[observations["compound_id"].isin(compound_ids)].copy().reset_index(drop=True)
        if self.frame.empty:
            raise ValueError("Dataset split is empty")
        self.cell_map = cell_map
        self.target_scaler = target_scaler
        self.context_prior = context_prior

        structures = self.frame[["compound_id", "canonical_smiles"]].drop_duplicates()
        self.fingerprints = {
            row.compound_id: morgan_fingerprint(
                row.canonical_smiles,
                radius=fingerprint_radius,
                use_counts=fingerprint_use_counts,
                include_chirality=fingerprint_include_chirality,
            )
            for row in structures.itertuples()
        }
        self.descriptors = {
            row.compound_id: molecular_descriptors(row.canonical_smiles, descriptor_set)
            for row in structures.itertuples()
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        baseline = np.asarray(row["baseline_expression"], dtype=np.float32)
        target_raw = np.asarray(row["delta_y"], dtype=np.float32)
        context_prior = self.context_prior.lookup(
            str(row["cell_line"]), float(row["dose_nM"]), float(row["duration_hours"])
        )
        target = self.target_scaler.transform(target_raw - context_prior).astype(np.float32)
        context = np.array(
            [
                (float(row["log10_dose_molar"]) + 6.5) / 1.5,
                np.log2(1.0 + float(row["duration_hours"])) / 7.0,
            ],
            dtype=np.float32,
        )
        compound_id = str(row["compound_id"])
        return {
            "sample_id": str(row["sample_id"]),
            "compound_id": compound_id,
            "fingerprint": torch.from_numpy(self.fingerprints[compound_id]),
            "descriptors": torch.from_numpy(self.descriptors[compound_id]),
            "cell_index": torch.tensor(self.cell_map[str(row["cell_line"])], dtype=torch.long),
            "context": torch.from_numpy(context),
            "baseline": torch.from_numpy(baseline),
            "target": torch.from_numpy(target),
            "target_raw": torch.from_numpy(target_raw),
            "context_prior": torch.from_numpy(context_prior),
            "replicate_pearson": torch.tensor(float(row["replicate_pearson"]), dtype=torch.float32),
        }


def build_datasets(
    observations: pd.DataFrame,
    manifest: dict,
    target_scaling: str = "per_gene",
    fingerprint_radius: int = 2,
    fingerprint_use_counts: bool = False,
    fingerprint_include_chirality: bool = False,
    descriptor_set: str = "basic",
) -> tuple[dict[str, SciPlexDataset], TargetScaler, dict[str, int], ContextPrior]:
    validate_split_manifest(observations, manifest)
    train_ids = set(manifest["splits"]["train"])
    train_frame = observations.loc[observations.compound_id.isin(train_ids)].copy()
    context_prior = ContextPrior.fit(train_frame)
    train_residuals = np.stack(
        [
            np.asarray(row.delta_y, dtype=np.float32)
            - context_prior.lookup(row.cell_line, row.dose_nM, row.duration_hours)
            for row in train_frame.itertuples()
        ]
    )
    scaler = TargetScaler.fit(train_residuals, mode=target_scaling)
    cell_map = {cell: index + 1 for index, cell in enumerate(sorted(observations.cell_line.unique()))}
    datasets = {
        name: SciPlexDataset(
            observations,
            set(manifest["splits"][name]),
            cell_map,
            scaler,
            context_prior,
            fingerprint_radius=fingerprint_radius,
            fingerprint_use_counts=fingerprint_use_counts,
            fingerprint_include_chirality=fingerprint_include_chirality,
            descriptor_set=descriptor_set,
        )
        for name in ("train", "validation", "test")
    }
    return datasets, scaler, cell_map, context_prior


def fit_response_basis(dataset: SciPlexDataset, rank: int | None) -> np.ndarray | None:
    """Fit a response basis from standardized training residuals only."""
    if rank is None or rank <= 0:
        return None
    targets = np.stack([dataset[index]["target"].numpy() for index in range(len(dataset))])
    maximum_rank = min(targets.shape)
    if rank >= maximum_rank:
        return None
    _, _, right_vectors = np.linalg.svd(targets, full_matrices=False)
    return right_vectors[:rank].astype(np.float32)
