from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from our_star.chemistry import (
    MolecularFormula,
    Reaction,
    ReactionExtentRate,
    ReactionNetwork,
    SpeciesFormula,
    StoichiometricTerm,
    UntrackedCoproduct,
    build_minimal_parent_metabolite_network,
    run_minimal_parent_metabolite_example,
)
from our_star.core import ChemicalSpecies


def test_parent_metabolite_example_conserves_molar_stoichiometry_and_mass() -> None:
    result = run_minimal_parent_metabolite_example()

    assert result.derivatives.species_umol_h == {
        "example_parent": -2.0,
        "example_metabolite": 2.0,
    }
    assert result.final_amounts_umol == {
        "example_parent": 4.0,
        "example_metabolite": 6.0,
    }
    assert result.cumulative_untracked_mass_mg == {
        "example_untracked_coproduct": pytest.approx(0.12)
    }

    initial_mass_mg = 10.0 * 200.0 / 1000.0
    final_tracked_mass_mg = 4.0 * 200.0 / 1000.0 + 6.0 * 180.0 / 1000.0
    final_total_mass_mg = final_tracked_mass_mg + sum(
        result.cumulative_untracked_mass_mg.values()
    )
    assert final_total_mass_mg == pytest.approx(initial_mass_mg, abs=1e-14)
    assert result.mass_accounting.tracked_mass_rate_mg_h == pytest.approx(-0.04)
    assert result.mass_accounting.untracked_coproduct_mass_rate_mg_h == pytest.approx(
        0.04
    )
    assert result.mass_accounting.closure_error_mg_h == pytest.approx(0.0, abs=1e-14)

    balance = result.network.balance_diagnostics[0]
    assert balance.elemental_balance_checked
    assert all(abs(value) < 1e-14 for value in balance.elemental_residuals.values())


def test_molecular_weight_difference_requires_declared_coproduct_mass() -> None:
    parent = ChemicalSpecies("parent", "parent", 200.0)
    metabolite = ChemicalSpecies("metabolite", "metabolite", 180.0)
    conversion = Reaction(
        "conversion",
        (
            StoichiometricTerm("parent", -1.0),
            StoichiometricTerm("metabolite", 1.0),
        ),
    )
    with pytest.raises(ValueError, match="not mass balanced"):
        ReactionNetwork((parent, metabolite), (conversion,))

    network = ReactionNetwork(
        (parent, metabolite),
        (
            Reaction(
                "conversion",
                conversion.terms,
                (
                    UntrackedCoproduct(
                        "lost_group", "declared molecular fragment", 20.0
                    ),
                ),
            ),
        ),
    )
    derivatives = network.derivatives((ReactionExtentRate("conversion", 5.0),))
    assert derivatives.species_umol_h["parent"] == -5.0
    assert derivatives.species_umol_h["metabolite"] == 5.0
    assert derivatives.total_untracked_coproduct_mg_h == pytest.approx(0.1)
    assert network.mass_accounting({"conversion": 5.0}).closure_error_mg_h == pytest.approx(
        0.0, abs=1e-14
    )


def test_invalid_reactions_and_elemental_accounting_fail_closed() -> None:
    parent = ChemicalSpecies("parent", "parent", 200.0)
    metabolite = ChemicalSpecies("metabolite", "metabolite", 180.0)
    coproduct = UntrackedCoproduct("fragment", "fragment", 20.0)

    with pytest.raises(ValueError, match="unknown species"):
        ReactionNetwork(
            (parent, metabolite),
            (
                Reaction(
                    "unknown",
                    (
                        StoichiometricTerm("parent", -1.0),
                        StoichiometricTerm("missing", 1.0),
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError, match="only once"):
        Reaction(
            "duplicate",
            (
                StoichiometricTerm("parent", -1.0),
                StoichiometricTerm("parent", 1.0),
            ),
        )
    with pytest.raises(ValueError, match="partial elemental formulas"):
        ReactionNetwork(
            (parent, metabolite),
            (
                Reaction(
                    "partial_formula",
                    (
                        StoichiometricTerm("parent", -1.0),
                        StoichiometricTerm("metabolite", 1.0),
                    ),
                    (coproduct,),
                ),
            ),
            species_formulas=(
                SpeciesFormula("parent", MolecularFormula.from_mapping({"C": 10})),
            ),
        )

    wrong_formula_coproduct = UntrackedCoproduct(
        "fragment",
        "fragment",
        20.0,
        formula=MolecularFormula.from_mapping({"H": 1}),
    )
    with pytest.raises(ValueError, match="not element balanced"):
        ReactionNetwork(
            (parent, metabolite),
            (
                Reaction(
                    "element_error",
                    (
                        StoichiometricTerm("parent", -1.0),
                        StoichiometricTerm("metabolite", 1.0),
                    ),
                    (wrong_formula_coproduct,),
                ),
            ),
            species_formulas=(
                SpeciesFormula("parent", MolecularFormula.from_mapping({"C": 10})),
                SpeciesFormula("metabolite", MolecularFormula.from_mapping({"C": 9})),
            ),
        )


def test_extent_kinetics_are_nonnegative_finite_and_sparse() -> None:
    network = build_minimal_parent_metabolite_network()
    with pytest.raises(ValueError, match=">= 0"):
        network.derivatives({"example_parent_to_metabolite": -1.0})
    with pytest.raises(ValueError, match="finite"):
        network.derivatives({"example_parent_to_metabolite": np.nan})
    with pytest.raises(ValueError, match="unknown reactions"):
        network.derivatives({"not_registered": 1.0})

    zero = network.derivatives({})
    assert zero.species_umol_h == {}
    assert zero.untracked_coproduct_mg_h == {}


def test_reaction_network_and_nested_outputs_are_immutable() -> None:
    network = build_minimal_parent_metabolite_network()
    with pytest.raises(FrozenInstanceError):
        network.reactions = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        network.derivatives({"example_parent_to_metabolite": 1.0}).species_umol_h[
            "example_parent"
        ] = 0.0  # type: ignore[index]


def test_example_refuses_extent_that_overconsumes_parent() -> None:
    with pytest.raises(ValueError, match="consume more parent"):
        run_minimal_parent_metabolite_example(
            initial_parent_umol=1.0,
            extent_umol_h=2.0,
            duration_h=1.0,
        )
