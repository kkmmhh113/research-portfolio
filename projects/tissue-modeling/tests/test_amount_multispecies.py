from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from our_star.chemistry.accounting import (
    DualAmountLedger,
    ExternalParticipant,
    ExternalParticipantRole,
    OpenReaction,
    OpenReactionNetwork,
    SpeciesMoiety,
)
from our_star.chemistry.reactions import StoichiometricTerm
from our_star.core.amount_state import (
    AmountStateRegistry,
    AmountStateSpec,
    FluxKey,
)
from our_star.core.amount_system import (
    AmountModelContext,
    AmountModuleResult,
    AmountSystemAssembler,
)
from our_star.core.species import ChemicalSpecies, SpeciesRegistry


def test_umol_state_registry_is_separate_from_v03_mg_registry() -> None:
    species = SpeciesRegistry((ChemicalSpecies("parent", "parent", 200.0),))
    registry = AmountStateRegistry(
        (AmountStateSpec("central::parent", "parent", "central"),), species
    )
    assert registry.amount_umol(np.asarray([4.0]), "central::parent") == 4.0
    assert registry.spec("central::parent").unit == "umol"
    with pytest.raises(ValueError, match="must use umol"):
        AmountStateSpec("wrong", "parent", "central", unit="mg")
    with pytest.raises(ValueError, match="unknown species"):
        AmountStateRegistry(
            (AmountStateSpec("central::missing", "missing", "central"),), species
        )


def test_open_mass_gaining_conjugation_and_dual_ledger_close() -> None:
    parent = ChemicalSpecies("parent", "parent", 200.0)
    conjugate = ChemicalSpecies("conjugate", "conjugate", 260.0)
    group = ExternalParticipant(
        "transfer_group",
        "synthetic transfer group",
        -1.0,
        60.0,
        ExternalParticipantRole.NET_TRANSFER_GROUP,
        provenance="synthetic unit-test chemistry",
    )
    network = OpenReactionNetwork(
        (parent, conjugate),
        (
            OpenReaction(
                "conjugation",
                (
                    StoichiometricTerm("parent", -1.0),
                    StoichiometricTerm("conjugate", 1.0),
                ),
                (group,),
                "synthetic mass-gaining conjugation",
            ),
        ),
    )
    diagnostic = network.mass_rate_diagnostic({"conjugation": 2.0})
    assert diagnostic.tracked_mass_rate_mg_h == pytest.approx(0.12)
    assert diagnostic.external_mass_rate_mg_h == pytest.approx(-0.12)
    assert diagnostic.closure_error_mg_h == pytest.approx(0.0, abs=1e-14)
    reaction_derivatives = network.derivatives({"conjugation": 2.0})
    assert reaction_derivatives.species_umol_h == {
        "parent": -2.0,
        "conjugate": 2.0,
    }
    assert reaction_derivatives.external_import_umol_h == {
        "transfer_group": 2.0
    }

    all_species = SpeciesRegistry(
        (
            parent,
            conjugate,
            ChemicalSpecies("transfer_group", "transfer group", 60.0),
        )
    )
    states = AmountStateRegistry(
        (
            AmountStateSpec("central::parent", "parent", "central"),
            AmountStateSpec("urine::conjugate", "conjugate", "urine", sink=True),
            AmountStateSpec(
                "external_import::transfer_group",
                "transfer_group",
                "external_import",
                sink=True,
                accounting_only=True,
            ),
        ),
        all_species,
    )
    ledger = DualAmountLedger(
        states,
        (
            SpeciesMoiety("parent", {"drug": 1.0}),
            SpeciesMoiety("conjugate", {"drug": 1.0}),
        ),
        conserved_moiety_id="drug",
        reference_species_id="parent",
        external_participants=network.external_participants,
        external_import_states={
            "transfer_group": "external_import::transfer_group"
        },
        external_export_states={},
    )
    report = ledger.report(np.asarray([4.0, 6.0, 6.0]))
    assert report.moiety_umol["drug"] == 10.0
    assert report.drug_equivalent_mass_mg == pytest.approx(2.0)
    assert report.tracked_molecular_mass_mg == pytest.approx(2.36)
    assert report.external_import_mass_mg == pytest.approx(0.36)
    assert report.augmented_accounted_mass_mg == pytest.approx(2.0)


@dataclass(frozen=True)
class _Producer:
    name: str = "producer"
    required_states: tuple[str, ...] = ("source::parent",)
    required_inputs: tuple[FluxKey, ...] = ()

    def evaluate(self, _t, amounts, _context, states, _inputs):
        amount = amounts[states.index("source::parent")]
        return AmountModuleResult(
            {"source::parent": -amount},
            {FluxKey("source_to_sink", "parent"): amount},
        )


@dataclass(frozen=True)
class _Consumer:
    name: str = "consumer"
    required_states: tuple[str, ...] = ("sink::parent",)
    required_inputs: tuple[FluxKey, ...] = (
        FluxKey("source_to_sink", "parent"),
    )

    def evaluate(self, _t, _amounts, _context, _states, inputs):
        return AmountModuleResult(
            {"sink::parent": inputs[FluxKey("source_to_sink", "parent")]}
        )


def test_species_aware_amount_assembler_conserves_connection_flux() -> None:
    registry = AmountStateRegistry(
        (
            AmountStateSpec("source::parent", "parent", "source"),
            AmountStateSpec("sink::parent", "parent", "sink"),
        ),
        SpeciesRegistry((ChemicalSpecies("parent", "parent", 100.0),)),
    )
    assembler = AmountSystemAssembler(registry, (_Producer(), _Consumer()))
    derivative = assembler.rhs(
        0.0,
        np.asarray([3.0, 0.0]),
        AmountModelContext(patient=None, drug=None),
    )
    assert derivative.tolist() == [-3.0, 3.0]
    assert derivative.sum() == pytest.approx(0.0)

    with pytest.raises(RuntimeError, match="missing connection flux"):
        AmountSystemAssembler(registry, (_Consumer(), _Producer())).rhs(
            0.0,
            np.asarray([3.0, 0.0]),
            AmountModelContext(patient=None, drug=None),
        )
