from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .chemistry import molecular_descriptors, molecular_graph, morgan_fingerprint


@dataclass
class MaskedTargetScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        targets: np.ndarray,
        masks: np.ndarray,
        mode: str = "global",
        minimum_scale: float = 0.05,
    ) -> "MaskedTargetScaler":
        targets = np.asarray(targets, dtype=np.float32)
        masks = np.asarray(masks, dtype=bool)
        if targets.shape != masks.shape or targets.ndim != 2:
            raise ValueError("Targets and masks must be matching two-dimensional arrays")
        counts = masks.sum(axis=0)
        observed_genes = counts > 0
        if not np.any(observed_genes):
            raise ValueError("Training split has no observed targets")
        mean = np.zeros(targets.shape[1], dtype=np.float32)
        mean[observed_genes] = np.divide(
            (targets * masks).sum(axis=0)[observed_genes], counts[observed_genes]
        )
        centered = targets - mean
        if mode == "global":
            scale_value = float(np.sqrt(np.square(centered)[masks].mean()))
            scale = np.full(targets.shape[1], scale_value, dtype=np.float32)
        elif mode == "per_gene":
            variance = np.zeros(targets.shape[1], dtype=np.float32)
            variance[observed_genes] = np.divide(
                (np.square(centered) * masks).sum(axis=0)[observed_genes],
                counts[observed_genes],
            )
            scale = np.sqrt(variance).astype(np.float32)
        else:
            raise ValueError(f"Unknown target scaling mode: {mode}")
        return cls(mean=mean, scale=np.maximum(scale, minimum_scale))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "MaskedTargetScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
        )


@dataclass
class MaskedContextPrior:
    values: dict[tuple[str, str, float, float], np.ndarray]
    fallback: np.ndarray

    @classmethod
    def fit(cls, observations: pd.DataFrame, output_dim: int) -> "MaskedContextPrior":
        targets = np.stack(observations["delta_y"].map(np.asarray)).astype(np.float32)
        masks = np.stack(observations["target_mask"].map(np.asarray)).astype(bool)
        fallback = cls._masked_mean(targets, masks, np.zeros(output_dim, dtype=np.float32))
        values: dict[tuple[str, float, float], np.ndarray] = {}
        frame = observations.copy()
        if "dataset_domain" not in frame:
            frame["dataset_domain"] = "unknown"
        for key, group in frame.groupby(
            ["dataset_domain", "cell_line", "dose_nM", "duration_hours"],
            sort=False,
        ):
            indices = group.index.to_numpy()
            values[
                (str(key[0]), str(key[1]), float(key[2]), float(key[3]))
            ] = cls._masked_mean(targets[indices], masks[indices], fallback)
        return cls(values=values, fallback=fallback)

    @staticmethod
    def _masked_mean(
        targets: np.ndarray,
        masks: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        counts = masks.sum(axis=0)
        result = np.asarray(fallback, dtype=np.float32).copy()
        observed = counts > 0
        result[observed] = np.divide(
            (targets * masks).sum(axis=0)[observed], counts[observed]
        )
        return result.astype(np.float32)

    def lookup(self, *args: str | float) -> np.ndarray:
        if len(args) == 3:
            domain, cell_line, dose_nm, duration_hours = "unknown", *args
        elif len(args) == 4:
            domain, cell_line, dose_nm, duration_hours = args
        else:
            raise TypeError("lookup expects cell/dose/time or domain/cell/dose/time")
        return self.values.get(
            (
                str(domain),
                str(cell_line),
                float(dose_nm),
                float(duration_hours),
            ),
            self.fallback,
        )

    def to_dict(self) -> dict:
        return {
            "values": {
                "\t".join([domain, cell, repr(dose), repr(duration)]): value.tolist()
                for (domain, cell, dose, duration), value in self.values.items()
            },
            "fallback": self.fallback.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MaskedContextPrior":
        values = {}
        for key, value in payload["values"].items():
            parts = key.split("\t")
            if len(parts) == 3:
                domain, cell, dose, duration = "unknown", *parts
            else:
                domain, cell, dose, duration = parts
            values[(domain, cell, float(dose), float(duration))] = np.asarray(
                value, dtype=np.float32
            )
        return cls(values=values, fallback=np.asarray(payload["fallback"], dtype=np.float32))


def load_prc_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).reset_index(drop=True)
    required = {
        "sample_id",
        "dataset_domain",
        "compound_id",
        "canonical_smiles",
        "cell_line",
        "dose_nM",
        "log10_dose_molar",
        "duration_hours",
        "delta_y",
        "target_mask",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"PRC dataset is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    dimensions = frame["delta_y"].map(lambda value: len(np.asarray(value))).unique()
    mask_dimensions = frame["target_mask"].map(lambda value: len(np.asarray(value))).unique()
    if len(dimensions) != 1 or not np.array_equal(dimensions, mask_dimensions):
        raise ValueError("Response and target-mask dimensions are inconsistent")
    return frame


def build_context_maps(paths: list[Path]) -> tuple[dict[str, int], dict[str, int]]:
    cells: set[str] = set()
    domains: set[str] = set()
    for path in paths:
        frame = pd.read_parquet(path, columns=["cell_line", "dataset_domain"])
        cells.update(frame["cell_line"].astype(str).unique())
        domains.update(frame["dataset_domain"].astype(str).unique())
    cell_map = {cell: index + 1 for index, cell in enumerate(sorted(cells))}
    domain_map = {domain: index for index, domain in enumerate(sorted(domains))}
    return cell_map, domain_map


def load_readout_features(
    path: Path,
    external_features_path: Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    panel = pd.read_csv(path).sort_values("common_panel_index").reset_index(drop=True)
    expected = np.arange(len(panel))
    if not np.array_equal(panel["common_panel_index"].to_numpy(), expected):
        raise ValueError("Readout panel indices must be contiguous and ordered")
    raw = np.column_stack(
        [
            panel["control_mean"].to_numpy(dtype=np.float32),
            np.log1p(panel["control_variance"].to_numpy(dtype=np.float32)),
            panel["control_detection_fraction"].to_numpy(dtype=np.float32),
        ]
    )
    mean = raw.mean(axis=0, keepdims=True)
    scale = np.maximum(raw.std(axis=0, keepdims=True), 1e-6)
    features = ((raw - mean) / scale).astype(np.float32)

    if external_features_path is not None:
        with np.load(external_features_path, allow_pickle=False) as archive:
            external = np.asarray(archive["features"], dtype=np.float32)
            gene_symbols = np.asarray(archive["gene_symbols"]).astype(str)
        expected_symbols = panel["gene_symbol"].astype(str).to_numpy()
        if external.ndim != 2 or external.shape[0] != len(panel):
            raise ValueError("External readout features do not match the panel length")
        if not np.array_equal(gene_symbols, expected_symbols):
            raise ValueError("External readout features are not aligned to the gene panel")
        if not np.isfinite(external).all():
            raise ValueError("External readout features contain non-finite values")
        features = np.concatenate([features, external], axis=1)
    return features.astype(np.float32), panel


def fit_compound_response_signatures(
    observations: pd.DataFrame,
    output_dim: int,
) -> dict[str, np.ndarray]:
    """Build scale-invariant train-only response directions for P pretraining."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for row in observations.itertuples():
        values = np.asarray(row.delta_y, dtype=np.float32).copy()
        mask = np.asarray(row.target_mask, dtype=bool)
        if values.shape != (output_dim,) or mask.shape != (output_dim,):
            raise ValueError("Response signature dimensions do not match the panel")
        if mask.sum() < 2:
            continue
        values[mask] -= values[mask].mean()
        norm = float(np.linalg.norm(values[mask]))
        if norm <= 1e-8:
            continue
        values[mask] /= norm
        values[~mask] = 0.0
        compound_id = str(row.compound_id)
        sums.setdefault(compound_id, np.zeros(output_dim, dtype=np.float32))
        counts.setdefault(compound_id, np.zeros(output_dim, dtype=np.float32))
        sums[compound_id] += values
        counts[compound_id] += mask

    signatures: dict[str, np.ndarray] = {}
    for compound_id, total in sums.items():
        observed = counts[compound_id] > 0
        signature = np.zeros(output_dim, dtype=np.float32)
        signature[observed] = total[observed] / counts[compound_id][observed]
        norm = float(np.linalg.norm(signature))
        if norm > 1e-8:
            signature /= norm
        signatures[compound_id] = signature
    return signatures


def parse_target_gene_symbols(value: object) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return sorted(
        {
            symbol.strip().upper()
            for symbol in str(value).split("|")
            if symbol.strip()
        }
    )


def parse_auxiliary_labels(value: object, uppercase: bool = False) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    labels = {
        label.strip().upper() if uppercase else label.strip()
        for label in str(value).split("|")
        if label.strip() and label.strip().casefold() not in {"nan", "none", "unclear"}
    }
    return sorted(labels, key=str.casefold)


def build_target_auxiliary_supervision(
    mechanism_supervision: pd.DataFrame | None,
    train_compound_ids: set[str],
    label_column: str = "specific_target_gene_symbols",
    minimum_compound_count: int = 1,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    if mechanism_supervision is None:
        return {}, {}
    required = {
        "compound_id",
        label_column,
        "allowed_as_inference_input",
    }
    missing = required.difference(mechanism_supervision.columns)
    if missing:
        raise ValueError(
            f"Mechanism supervision is missing columns: {sorted(missing)}"
        )
    if mechanism_supervision["compound_id"].duplicated().any():
        raise ValueError("Mechanism supervision must contain one row per compound")
    if mechanism_supervision["allowed_as_inference_input"].astype(bool).any():
        raise ValueError("Mechanism labels must remain unavailable at inference")
    if minimum_compound_count <= 0:
        raise ValueError("minimum_compound_count must be positive")

    train_rows = mechanism_supervision[
        mechanism_supervision["compound_id"].astype(str).isin(train_compound_ids)
    ].copy()
    if label_column == "specific_target_gene_symbols":
        required_eligibility = "eligible_for_target_auxiliary_training"
        if required_eligibility not in train_rows:
            raise ValueError(
                f"Mechanism supervision is missing column: {required_eligibility}"
            )
        train_rows = train_rows[
            train_rows[required_eligibility].astype(bool)
        ]
    uppercase = label_column == "specific_target_gene_symbols"
    label_counts: dict[str, int] = {}
    for value in train_rows[label_column]:
        for label in parse_auxiliary_labels(value, uppercase=uppercase):
            label_counts[label] = label_counts.get(label, 0) + 1
    vocabulary = sorted(
        label
        for label, count in label_counts.items()
        if count >= minimum_compound_count
    )
    target_gene_map = {symbol: index for index, symbol in enumerate(vocabulary)}
    labels: dict[str, np.ndarray] = {}
    if not target_gene_map:
        return target_gene_map, labels
    for record in mechanism_supervision.to_dict("records"):
        symbols = parse_auxiliary_labels(
            record[label_column], uppercase=uppercase
        )
        vector = np.zeros(len(target_gene_map), dtype=np.float32)
        for symbol in symbols:
            index = target_gene_map.get(symbol)
            if index is not None:
                vector[index] = 1.0
        if vector.any():
            labels[str(record["compound_id"])] = vector
    return target_gene_map, labels


class PRCDataset(Dataset):
    def __init__(
        self,
        observations: pd.DataFrame,
        compound_ids: set[str],
        cell_map: dict[str, int],
        domain_map: dict[str, int],
        scaler: MaskedTargetScaler,
        context_prior: MaskedContextPrior,
        response_signatures: dict[str, np.ndarray] | None = None,
        pathway_map: dict[str, int] | None = None,
        target_gene_map: dict[str, int] | None = None,
        compound_target_labels: dict[str, np.ndarray] | None = None,
        descriptor_set: str = "extended",
        fingerprint_radius: int = 2,
        fingerprint_use_counts: bool = True,
        fingerprint_include_chirality: bool = True,
        chemical_encoder_mode: str = "ecfp",
    ) -> None:
        frame = observations[observations["compound_id"].isin(compound_ids)].copy()
        self.frame = frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError("PRC dataset split is empty")
        self.cell_map = cell_map
        self.domain_map = domain_map
        self.scaler = scaler
        self.context_prior = context_prior
        self.response_signatures = response_signatures or {}
        self.pathway_map = pathway_map or {}
        self.target_gene_map = target_gene_map or {}
        self.compound_target_labels = compound_target_labels or {}
        structures = self.frame[["compound_id", "canonical_smiles"]].drop_duplicates()
        if structures["compound_id"].duplicated().any():
            raise ValueError("A compound ID maps to multiple structures")
        if chemical_encoder_mode not in {"ecfp", "graph"}:
            raise ValueError(f"Unknown chemical encoder mode: {chemical_encoder_mode}")
        self.chemical_encoder_mode = chemical_encoder_mode
        self.chemical_permutation: dict[str, str] | None = None
        self.fingerprints = (
            {
                str(row.compound_id): morgan_fingerprint(
                    str(row.canonical_smiles),
                    radius=fingerprint_radius,
                    use_counts=fingerprint_use_counts,
                    include_chirality=fingerprint_include_chirality,
                )
                for row in structures.itertuples()
            }
            if chemical_encoder_mode == "ecfp"
            else {}
        )
        self.graphs = (
            {
                str(row.compound_id): molecular_graph(str(row.canonical_smiles))
                for row in structures.itertuples()
            }
            if chemical_encoder_mode == "graph"
            else {}
        )
        self.descriptors = {
            str(row.compound_id): molecular_descriptors(
                str(row.canonical_smiles), descriptor_set=descriptor_set
            )
            for row in structures.itertuples()
        }
        self.targets = np.stack(self.frame["delta_y"].map(np.asarray)).astype(np.float32)
        self.masks = np.stack(self.frame["target_mask"].map(np.asarray)).astype(bool)
        self.output_dim = self.targets.shape[1]
        if "target_std" in self.frame:
            self.target_stds = np.stack(
                self.frame["target_std"].map(np.asarray)
            ).astype(np.float32)
            self.target_std_masks = np.isfinite(self.target_stds) & (self.target_stds >= 0)
            self.target_stds = np.nan_to_num(self.target_stds, nan=0.0)
        else:
            self.target_stds = np.zeros_like(self.targets)
            self.target_std_masks = np.zeros_like(self.masks)
        if "baseline_expression" in self.frame:
            baselines = []
            for row in self.frame.itertuples():
                baseline = np.asarray(row.baseline_expression, dtype=np.float32)
                if baseline.shape == (self.output_dim,):
                    baselines.append(baseline)
                elif not bool(getattr(row, "baseline_available", False)):
                    baselines.append(np.zeros(self.output_dim, dtype=np.float32))
                else:
                    raise ValueError(
                        f"Available baseline has invalid shape for sample {row.sample_id}"
                    )
            self.baselines = np.stack(baselines).astype(np.float32)
        else:
            self.baselines = np.zeros_like(self.targets)
        if self.baselines.shape != self.targets.shape:
            raise ValueError("Baseline and response dimensions do not match")

        priors = []
        for row in self.frame.itertuples():
            priors.append(
                context_prior.lookup(
                    row.dataset_domain,
                    row.cell_line,
                    row.dose_nM,
                    row.duration_hours,
                )
            )
        self.priors = np.stack(priors).astype(np.float32)
        self.scaled_targets = scaler.transform(self.targets - self.priors).astype(np.float32)
        self.scaled_target_stds = (self.target_stds / scaler.scale).astype(np.float32)
        self.domain_labels = self.frame["dataset_domain"].astype(str).to_numpy()
        active_target_compounds = sorted(
            set(self.frame["compound_id"].astype(str)).intersection(
                self.compound_target_labels
            )
        )
        self.target_label_compound_count = len(active_target_compounds)
        if active_target_compounds:
            unique_labels = np.stack(
                [self.compound_target_labels[value] for value in active_target_compounds]
            )
            positives = unique_labels.sum(axis=0)
            negatives = len(unique_labels) - positives
            self.target_positive_weights = np.clip(
                negatives / np.maximum(positives, 1.0), 1.0, 20.0
            ).astype(np.float32)
        else:
            self.target_positive_weights = np.ones(
                len(self.target_gene_map), dtype=np.float32
            )

    def __len__(self) -> int:
        return len(self.frame)

    def set_chemical_permutation(self, mapping: dict[str, str] | None) -> None:
        if mapping is not None:
            available = set(self.descriptors)
            if set(mapping) != available or not set(mapping.values()).issubset(available):
                raise ValueError("Chemical permutation must cover the split compounds exactly")
        self.chemical_permutation = mapping

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        compound_id = str(row["compound_id"])
        feature_compound_id = (
            self.chemical_permutation.get(compound_id, compound_id)
            if self.chemical_permutation is not None
            else compound_id
        )
        domain = str(row["dataset_domain"])
        baseline_available = float(row.get("baseline_available", 0.0))
        numeric_context = np.asarray(
            [
                (float(row["log10_dose_molar"]) + 6.5) / 1.5,
                np.log2(1.0 + float(row["duration_hours"])) / 7.0,
                baseline_available,
            ],
            dtype=np.float32,
        )
        result = {
            "sample_id": str(row["sample_id"]),
            "compound_id": compound_id,
            "fingerprint": torch.from_numpy(
                self.fingerprints.get(
                    feature_compound_id, np.zeros(2048, dtype=np.float32)
                )
            ),
            "descriptors": torch.from_numpy(self.descriptors[feature_compound_id]),
            "cell_index": torch.tensor(
                self.cell_map.get(str(row["cell_line"]), 0), dtype=torch.long
            ),
            "domain_index": torch.tensor(self.domain_map[domain], dtype=torch.long),
            "context": torch.from_numpy(numeric_context),
            "baseline": torch.from_numpy(self.baselines[index]),
            "target": torch.from_numpy(self.scaled_targets[index]),
            "target_raw": torch.from_numpy(self.targets[index]),
            "target_mask": torch.from_numpy(self.masks[index]),
            "target_std": torch.from_numpy(self.scaled_target_stds[index]),
            "target_std_mask": torch.from_numpy(self.target_std_masks[index]),
            "context_prior": torch.from_numpy(self.priors[index]),
            "response_signature": torch.from_numpy(
                self.response_signatures.get(
                    compound_id, np.zeros(self.output_dim, dtype=np.float32)
                )
            ),
            "response_signature_available": torch.tensor(
                float(compound_id in self.response_signatures), dtype=torch.float32
            ),
            "pathway_index": torch.tensor(
                self.pathway_map.get(str(row.get("pathway_annotation", "")), 0),
                dtype=torch.long,
            ),
            "target_gene_labels": torch.from_numpy(
                self.compound_target_labels.get(
                    compound_id,
                    np.zeros(len(self.target_gene_map), dtype=np.float32),
                )
            ),
            "target_gene_labels_available": torch.tensor(
                float(compound_id in self.compound_target_labels), dtype=torch.float32
            ),
            "sample_weight": torch.tensor(
                float(row.get("sample_weight", 1.0)), dtype=torch.float32
            ),
        }
        if self.chemical_encoder_mode == "graph":
            atom_features, bond_types = self.graphs[feature_compound_id]
            result["atom_features"] = torch.from_numpy(atom_features)
            result["bond_types"] = torch.from_numpy(bond_types)
        return result


def collate_prc_batch(samples: list[dict]) -> dict:
    """Collate PRC records while dynamically padding molecular graphs."""
    if not samples or "atom_features" not in samples[0]:
        return default_collate(samples)
    copied = [dict(sample) for sample in samples]
    atom_features = [sample.pop("atom_features") for sample in copied]
    bond_types = [sample.pop("bond_types") for sample in copied]
    batch = default_collate(copied)
    maximum_atoms = max(values.shape[0] for values in atom_features)
    padded_features = torch.zeros(
        len(samples), maximum_atoms, 7, dtype=torch.long
    )
    padded_bonds = torch.zeros(
        len(samples), maximum_atoms, maximum_atoms, dtype=torch.long
    )
    atom_mask = torch.zeros(len(samples), maximum_atoms, dtype=torch.bool)
    for index, (features, bonds) in enumerate(zip(atom_features, bond_types)):
        atoms = features.shape[0]
        padded_features[index, :atoms] = features
        padded_bonds[index, :atoms, :atoms] = bonds
        atom_mask[index, :atoms] = True
    batch["atom_features"] = padded_features
    batch["bond_types"] = padded_bonds
    batch["atom_mask"] = atom_mask
    return batch


def build_stage_datasets(
    observations: pd.DataFrame,
    split_manifest: dict,
    cell_map: dict[str, int],
    domain_map: dict[str, int],
    target_scaling: str,
    descriptor_set: str,
    fingerprint_radius: int,
    fingerprint_use_counts: bool,
    fingerprint_include_chirality: bool,
    include_validation: bool = True,
    mechanism_supervision: pd.DataFrame | None = None,
    target_auxiliary_label_column: str = "specific_target_gene_symbols",
    target_auxiliary_min_compounds: int = 1,
    chemical_encoder_mode: str = "ecfp",
) -> tuple[dict[str, PRCDataset], MaskedTargetScaler, MaskedContextPrior]:
    train_ids = set(split_manifest["splits"]["train"])
    validation_ids = set(split_manifest["splits"].get("validation", []))
    if train_ids.intersection(validation_ids):
        raise ValueError("Compound leakage between PRC train and validation splits")
    if not train_ids:
        raise ValueError("PRC training split is empty")
    if include_validation and not validation_ids:
        raise ValueError("PRC validation split is empty")
    available = set(observations["compound_id"].unique())
    if not train_ids.union(validation_ids).issubset(available):
        raise ValueError("PRC split refers to compounds absent from the dataset")
    train_frame = observations[observations["compound_id"].isin(train_ids)].copy()
    output_dim = len(np.asarray(train_frame.iloc[0]["delta_y"]))
    prior = MaskedContextPrior.fit(train_frame.reset_index(drop=True), output_dim)
    targets = np.stack(train_frame["delta_y"].map(np.asarray)).astype(np.float32)
    masks = np.stack(train_frame["target_mask"].map(np.asarray)).astype(bool)
    priors = np.stack(
        [
            prior.lookup(
                row.dataset_domain,
                row.cell_line,
                row.dose_nM,
                row.duration_hours,
            )
            for row in train_frame.itertuples()
        ]
    )
    scaler = MaskedTargetScaler.fit(targets - priors, masks, mode=target_scaling)
    response_signatures = fit_compound_response_signatures(train_frame, output_dim)
    pathway_values = sorted(
        value
        for value in train_frame.get("pathway_annotation", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        if value.strip()
    )
    pathway_map = {value: index + 1 for index, value in enumerate(pathway_values)}
    target_gene_map, compound_target_labels = build_target_auxiliary_supervision(
        mechanism_supervision,
        train_ids,
        label_column=target_auxiliary_label_column,
        minimum_compound_count=target_auxiliary_min_compounds,
    )
    kwargs = {
        "observations": observations,
        "cell_map": cell_map,
        "domain_map": domain_map,
        "scaler": scaler,
        "context_prior": prior,
        "response_signatures": response_signatures,
        "pathway_map": pathway_map,
        "target_gene_map": target_gene_map,
        "compound_target_labels": compound_target_labels,
        "descriptor_set": descriptor_set,
        "fingerprint_radius": fingerprint_radius,
        "fingerprint_use_counts": fingerprint_use_counts,
        "fingerprint_include_chirality": fingerprint_include_chirality,
        "chemical_encoder_mode": chemical_encoder_mode,
    }
    datasets = {"train": PRCDataset(compound_ids=train_ids, **kwargs)}
    if include_validation:
        datasets["validation"] = PRCDataset(compound_ids=validation_ids, **kwargs)
    return datasets, scaler, prior


def _fit_masked_response_basis_arrays(
    targets: np.ndarray, masks: np.ndarray, rank: int
) -> np.ndarray | None:
    if rank <= 0:
        return None
    targets = np.asarray(targets, dtype=np.float32).copy()
    masks = np.asarray(masks, dtype=bool)
    if targets.shape != masks.shape or targets.ndim != 2:
        raise ValueError("Targets and masks must be matching two-dimensional arrays")
    targets[~masks] = 0.0
    maximum_rank = min(targets.shape)
    if rank >= maximum_rank:
        return None
    covariance = targets.T @ targets / max(len(targets), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    selected = np.argsort(eigenvalues)[-rank:][::-1]
    basis = eigenvectors[:, selected].T.astype(np.float32)
    if not np.isfinite(basis).all():
        raise ValueError("Response basis contains non-finite values")
    return basis


def fit_masked_response_basis(dataset: PRCDataset, rank: int) -> np.ndarray | None:
    """Fit orthonormal gene loadings from one training dataset's residuals."""
    return _fit_masked_response_basis_arrays(
        dataset.scaled_targets, dataset.masks, rank
    )


def fit_shared_response_basis(
    observations: pd.DataFrame,
    split_manifest: dict,
    target_scaling: str,
    rank: int,
) -> np.ndarray | None:
    """Fit a reusable basis from a designated train split without validation access."""
    train_ids = set(split_manifest["splits"]["train"])
    validation_ids = set(split_manifest["splits"]["validation"])
    if train_ids.intersection(validation_ids):
        raise ValueError("Compound leakage in shared response-basis split")
    train_frame = observations[observations["compound_id"].isin(train_ids)].copy()
    if train_frame.empty:
        raise ValueError("Shared response-basis training split is empty")
    output_dim = len(np.asarray(train_frame.iloc[0]["delta_y"]))
    prior = MaskedContextPrior.fit(train_frame.reset_index(drop=True), output_dim)
    targets = np.stack(train_frame["delta_y"].map(np.asarray)).astype(np.float32)
    masks = np.stack(train_frame["target_mask"].map(np.asarray)).astype(bool)
    priors = np.stack(
        [
            prior.lookup(
                getattr(row, "dataset_domain", "unknown"),
                row.cell_line,
                row.dose_nM,
                row.duration_hours,
            )
            for row in train_frame.itertuples()
        ]
    )
    scaler = MaskedTargetScaler.fit(targets - priors, masks, mode=target_scaling)
    scaled_targets = scaler.transform(targets - priors).astype(np.float32)
    return _fit_masked_response_basis_arrays(scaled_targets, masks, rank)
