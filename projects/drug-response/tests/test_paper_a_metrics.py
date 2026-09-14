from dataclasses import replace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from scipy.stats import bootstrap, pearsonr

from my_star.paper_a_metrics import (
    PearsonPolicy, Predictions, Reference, compound_macro_bootstrap,
    equal_seed_ensemble, evaluate_paired,
)


POLICY = PearsonPolicy(degenerate="zero_with_flag", denominator_floor=1e-12)
A = np.array([1., -1., 0.]) / np.sqrt(2)
B = np.array([1., 1., -2.]) / np.sqrt(6)


def fixture(spec=None):
    """Artificial correlations for calculation tests, never measured responses."""
    spec = spec or [("drug-a", 1, 0.1)] * 9 + [("drug-b", 2, -0.5)]
    metadata = pd.DataFrame([dict(sample_id=f"sample-{i:03}", compound_id=c, fold_id=f,
                                  cell_line="SYNTHETIC", dose_nM=float(i + 1), duration_hours=24.)
                             for i, (c, f, _) in enumerate(spec)])
    target = np.tile(A, (len(spec), 1))
    predicted = np.stack([r * A + np.sqrt(1 - r*r) * B for _, _, r in spec])
    genes = ("synthetic-gene-1", "synthetic-gene-2", "synthetic-gene-3")
    reference = Reference(metadata, genes, target, np.ones(target.shape, dtype=bool))
    return reference, Predictions(metadata.copy(), genes, predicted), Predictions(metadata.copy(), genes, np.tile(B, (len(spec), 1)))


def evaluate(reference, candidate, baseline, **kwargs):
    options = dict(allowed_folds=(1, 2), pearson_policy=POLICY, iterations=1000, bootstrap_seed=2026)
    return evaluate_paired(reference, candidate, baseline, **(options | kwargs))


def permute(block, order):
    field = "targets" if isinstance(block, Reference) else "values"
    updates = {"metadata": block.metadata.iloc[order], field: getattr(block, field)[order]}
    if isinstance(block, Reference):
        updates["mask"] = block.mask[order]
    return replace(block, **updates)


def test_hand_calculation_macro_and_row_micro_have_opposite_signs():
    report, rows, compounds = evaluate(*fixture())
    assert report["primary"]["observed_mean_difference"] == pytest.approx(-0.2)
    assert report["secondary_row_micro_paired_gain"] == pytest.approx(0.04)
    assert compounds.conditions.tolist() == [9, 1]
    assert compounds.paired_gain.tolist() == pytest.approx([0.1, -0.5])
    assert len(rows) == 10 and report["promotion_or_access_approval_conferred"] is False


def test_historical_cluster_bootstrap_is_not_the_new_macro_estimand():
    from scripts.analyze_prc_phase8_confirmation import compound_block_bootstrap
    old = compound_block_bootstrap({"drug-a": [0.1] * 9, "drug-b": [-0.5]}, 1000, seed=2026)
    new = compound_macro_bootstrap({"drug-a": 0.1, "drug-b": -0.5}, iterations=1000)
    assert old["observed_mean_difference"] == pytest.approx(0.04)
    assert new["observed_mean_difference"] == pytest.approx(-0.2)


def test_extra_conditions_do_not_reweight_a_compound():
    balanced = fixture([("drug-a", 1, 0.1), ("drug-b", 2, -0.5)])
    first = evaluate(*balanced)[0]
    repeated = evaluate(*fixture())[0]
    assert first["primary"]["observed_mean_difference"] == pytest.approx(repeated["primary"]["observed_mean_difference"])
    assert first["primary"]["ci95_lower"] == pytest.approx(repeated["primary"]["ci95_lower"])
    assert first["secondary_row_micro_paired_gain"] != pytest.approx(repeated["secondary_row_micro_paired_gain"])


def test_compounds_not_folds_have_equal_weight_in_pooled_confirmation():
    report, _, _ = evaluate(*fixture([("a", 1, .5), ("b", 1, .5), ("c", 2, -.5)]))
    assert report["primary"]["observed_mean_difference"] == pytest.approx(1 / 6)
    assert report["per_fold"]["1"]["compound_macro_paired_gain"] == pytest.approx(.5)
    assert report["per_fold"]["2"]["compound_macro_paired_gain"] == pytest.approx(-.5)


def test_row_order_cannot_change_pairing_or_results():
    reference, candidate, baseline = fixture()
    report, rows, compounds = evaluate(reference, candidate, baseline)
    actual = evaluate(permute(reference, list(range(9, -1, -1))),
                      permute(candidate, [9, 1, 2, 3, 4, 5, 6, 7, 8, 0]), baseline)
    assert actual[0] == report
    pdt.assert_frame_equal(actual[1], rows)
    pdt.assert_frame_equal(actual[2], compounds)


def test_masked_pearson_matches_independent_scipy_and_ignores_only_unobserved_values():
    reference, candidate, baseline = fixture()
    mask = reference.mask.copy()
    mask[::2, 2] = False
    target, values, prior = reference.targets.copy(), candidate.values.copy(), baseline.values.copy()
    target[~mask], values[~mask], prior[~mask] = np.nan, np.inf, -np.inf
    report, rows, _ = evaluate(replace(reference, targets=target, mask=mask),
                               replace(candidate, values=values), replace(baseline, values=prior))
    for i, row in rows.iterrows():
        assert row.candidate_pearson_score == pytest.approx(pearsonr(target[i, mask[i]], values[i, mask[i]]).statistic)
    assert report["conditions"] == len(reference.metadata)


def test_bootstrap_matches_independent_scalar_reference_and_scipy_percentile():
    effects = {"d": .3, "a": -.5, "c": .1, "b": .8}
    report = compound_macro_bootstrap(effects, iterations=2000, seed=19)
    values = np.array([effects[k] for k in sorted(effects)])
    generator = np.random.Generator(np.random.PCG64(19))
    expected = [np.mean(values[generator.integers(len(values), size=len(values))]) for _ in range(2000)]
    lo, hi = np.quantile(expected, [.025, .975], method="linear")
    external = bootstrap((values,), np.mean, method="percentile", n_resamples=2000,
                         rng=np.random.Generator(np.random.PCG64(19)))
    assert report["ci95_lower"] == lo == external.confidence_interval.low
    assert report["ci95_upper"] == hi == external.confidence_interval.high
    assert report["bootstrap_mean_difference"] == np.mean(expected)
    assert report["bootstrap_fraction_above_zero"] == np.mean(np.asarray(expected) > 0)
    assert report["positive_fraction_is_p_value_or_posterior_probability"] is False
    assert report == compound_macro_bootstrap(dict(reversed(list(effects.items()))), iterations=2000, seed=19)


def test_identical_methods_preserve_pairing_and_zero_uncertainty():
    reference, candidate, _ = fixture()
    report, _, _ = evaluate(reference, candidate, candidate)
    primary = report["primary"]
    assert primary["observed_mean_difference"] == primary["ci95_lower"] == primary["ci95_upper"] == 0
    assert primary["bootstrap_fraction_above_zero"] == 0
    assert primary["bootstrap_fraction_equal_zero"] == 1
    assert primary["degenerate_bootstrap_distribution"] is True


def test_seed_predictions_are_averaged_before_nonlinear_scoring():
    reference, candidate, baseline = fixture()
    vectors = {2026: 2*A+B, 3407: -A, 9001: B}
    members = {s: replace(candidate, values=np.tile(v, (len(reference.metadata), 1))) for s, v in vectors.items()}
    members[3407] = permute(members[3407], list(range(9, -1, -1)))
    ensemble = equal_seed_ensemble(reference, members, required_seeds=(9001, 3407, 2026))
    report, _, _ = evaluate(reference, ensemble, baseline)
    expected = pearsonr(A, (A+2*B)/3).statistic
    average_scores = np.mean([pearsonr(A, v).statistic for v in vectors.values()])
    assert report["compound_macro_candidate_score"] == pytest.approx(expected)
    assert report["compound_macro_candidate_score"] != pytest.approx(average_scores)


@pytest.mark.parametrize("mutation", ["missing", "extra", "bool_key"])
def test_wrong_seed_population_is_rejected(mutation):
    reference, candidate, _ = fixture()
    members = {2026: candidate, 3407: candidate, 9001: candidate}
    if mutation == "missing":
        del members[3407]
    elif mutation == "extra":
        members[13] = candidate
    else:
        members[True] = members.pop(2026)
    with pytest.raises(ValueError, match="seed member"):
        equal_seed_ensemble(reference, members, required_seeds=(2026, 3407, 9001))


@pytest.mark.parametrize("field,value", [("sample_id", "changed"), ("compound_id", "changed"),
                                        ("fold_id", 9), ("cell_line", "changed"),
                                        ("dose_nM", 99.), ("duration_hours", 12.)])
def test_prediction_metadata_mismatch_is_fatal(field, value):
    reference, candidate, baseline = fixture()
    metadata = candidate.metadata.copy()
    metadata.loc[0, field] = value
    with pytest.raises(ValueError):
        evaluate(reference, replace(candidate, metadata=metadata), baseline)


@pytest.mark.parametrize("mutation", ["duplicate_id", "duplicate_condition", "compound_cross_fold", "blank_id",
                                      "invalid_fold", "nan_dose", "nullable_nan_dose", "negative_duration"])
def test_reference_population_is_not_silently_repaired(mutation):
    reference, candidate, baseline = fixture()
    metadata = reference.metadata.copy()
    if mutation == "duplicate_id":
        metadata.loc[0, "sample_id"] = metadata.loc[1, "sample_id"]
    elif mutation == "duplicate_condition":
        metadata.loc[0, "dose_nM"] = metadata.loc[1, "dose_nM"]
    elif mutation == "compound_cross_fold":
        metadata.loc[0, "fold_id"] = 2
    elif mutation == "blank_id":
        metadata.loc[0, "compound_id"] = " "
    elif mutation == "invalid_fold":
        metadata["fold_id"] = True
    elif mutation == "nan_dose":
        metadata.loc[0, "dose_nM"] = np.nan
    elif mutation == "nullable_nan_dose":
        metadata["dose_nM"] = metadata.dose_nM.astype("Float64")
        metadata.loc[0, "dose_nM"] = pd.NA
    else:
        metadata.loc[0, "duration_hours"] = -1.
    with pytest.raises(ValueError):
        evaluate(replace(reference, metadata=metadata), candidate, baseline)


@pytest.mark.parametrize("mutation", ["missing_sample", "wrong_gene_order", "wrong_gene_identity", "wrong_shape", "nan_observed"])
def test_prediction_population_gene_axis_and_observed_values_must_match(mutation):
    reference, candidate, baseline = fixture()
    if mutation == "missing_sample":
        candidate = replace(candidate, metadata=candidate.metadata.iloc[:-1], values=candidate.values[:-1])
    elif mutation == "wrong_gene_order":
        candidate = replace(candidate, gene_ids=candidate.gene_ids[::-1], values=candidate.values[:, ::-1])
    elif mutation == "wrong_gene_identity":
        candidate = replace(candidate, gene_ids=("other-gene", *candidate.gene_ids[1:]))
    elif mutation == "wrong_shape":
        candidate = replace(candidate, values=candidate.values[:, :-1])
    else:
        values = candidate.values.copy()
        values[0, 0] = np.nan
        candidate = replace(candidate, values=values)
    with pytest.raises(ValueError):
        evaluate(reference, candidate, baseline)


@pytest.mark.parametrize("mutation", ["integer_mask", "insufficient_genes", "nan_target", "duplicate_gene", "unexpected_fold"])
def test_reference_axis_mask_and_declared_folds_are_explicit(mutation):
    reference, candidate, baseline = fixture()
    options = {}
    if mutation == "integer_mask":
        reference = replace(reference, mask=reference.mask.astype(int))
    elif mutation == "insufficient_genes":
        mask = reference.mask.copy()
        mask[0, :2] = False
        reference = replace(reference, mask=mask)
    elif mutation == "nan_target":
        values = reference.targets.copy()
        values[0, 0] = np.nan
        reference = replace(reference, targets=values)
    elif mutation == "duplicate_gene":
        reference = replace(reference, gene_ids=("same", "same", "other"))
    else:
        options = {"allowed_folds": (0, 1, 2)}
    with pytest.raises(ValueError):
        evaluate(reference, candidate, baseline, **options)


def test_constant_predictions_are_flagged_or_rejected_never_silently_discarded():
    reference, candidate, baseline = fixture()
    zero = replace(baseline, values=np.zeros_like(baseline.values))
    report, rows, _ = evaluate(reference, candidate, zero)
    assert report["degenerate_rows"]["baseline"] == len(reference.metadata)
    assert len(rows) == len(reference.metadata) and (rows.baseline_pearson_score == 0).all()
    assert report["zero_fallback_is_mathematical_Pearson"] is False
    with pytest.raises(ValueError, match="Undefined/degenerate"):
        evaluate(reference, candidate, zero, pearson_policy=PearsonPolicy("error", 1e-12))


@pytest.mark.parametrize("effects,options", [({}, {}), ({"a": .1}, {}), ({"a": np.nan, "b": .2}, {}),
    ({"a": True, "b": .2}, {}), ({"a": .1, "b": .2}, {"iterations": 0}),
    ({"a": .1, "b": .2}, {"seed": True})])
def test_invalid_bootstrap_does_not_produce_a_confidence_claim(effects, options):
    with pytest.raises(ValueError):
        compound_macro_bootstrap(effects, **options)


def test_synthetic_audit_does_not_read_dataset_tables(monkeypatch):
    from scripts.audit_paper_a_metrics import audit
    def forbidden(*args, **kwargs):
        pytest.fail("Dataset table read during synthetic-only audit")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    report = audit()
    assert report["dataset_or_prediction_files_read"] is False
    assert report["calculated"]["compound_macro"] == pytest.approx(-.2)


def test_synthetic_audit_cli_refuses_dataset_option_and_overwrite(tmp_path, monkeypatch):
    from scripts import audit_paper_a_metrics as audit_script
    monkeypatch.setattr("sys.argv", ["audit", "--dataset", "forbidden.parquet"])
    with pytest.raises(SystemExit) as error:
        audit_script.main()
    assert error.value.code == 2
    path = tmp_path / "existing.json"
    path.write_text("preserve")
    monkeypatch.setattr(audit_script, "OUTPUT", path)
    monkeypatch.setattr("sys.argv", ["audit"])
    with pytest.raises(FileExistsError):
        audit_script.main()
    assert path.read_text() == "preserve"
