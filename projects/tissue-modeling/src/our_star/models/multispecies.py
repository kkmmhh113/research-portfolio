"""Generic simulation adapter for µmol, multi-species PBPK assemblies."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ..chemistry.accounting import DualAmountLedger, OpenReactionNetwork
from ..core.amount_state import AmountStateRegistry, SpeciesDoseEvent
from ..core.amount_system import AmountModelContext, AmountSystemAssembler
from ..core.species import AmountUnit
from ..pbpk import DoseEvent, DrugParameters, PatientPhysiology


class MultiSpeciesPBPKModel:
    """Reusable bridge from a µmol assembly to the historical model protocol.

    ``virtual_trial.simulate_patient`` calls the same four methods for old and
    new models.  For this class the returned solver vector is in µmol despite
    the legacy result container's ``states_mg`` field name.  Consumers should
    inspect ``state_amount_unit`` and use the named concentration/accounting
    record rather than treating raw columns as mass.
    """

    state_amount_unit = AmountUnit.UMOL

    def __init__(
        self,
        *,
        model_id: str,
        drug: DrugParameters,
        formulation: object,
        states: AmountStateRegistry,
        assembler: AmountSystemAssembler,
        reaction_network: OpenReactionNetwork,
        ledger: DualAmountLedger,
        parent_species_id: str,
        dose_targets: Mapping[tuple[str, str], str],
        central_states: Mapping[str, str],
        urine_states: Mapping[str, str],
        feces_parent_state: str,
        parent_liver_states: Sequence[str] = (),
        mechanism_factors: Mapping[str, float] | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("multi-species model_id is required")
        if parent_species_id not in states.species:
            raise ValueError("parent species is not registered")
        if parent_species_id not in central_states:
            raise ValueError("central_states must include the parent species")
        all_referenced_states = (
            set(dose_targets.values())
            | set(central_states.values())
            | set(urine_states.values())
            | {feces_parent_state}
            | set(parent_liver_states)
        )
        unknown_states = sorted(all_referenced_states - set(states.names))
        if unknown_states:
            raise ValueError(
                f"multi-species model references unknown states: {unknown_states}"
            )
        for (route, species_id), state_name in dose_targets.items():
            if route not in {"oral", "iv_bolus"}:
                raise ValueError(f"unsupported multi-species dose route: {route}")
            if species_id not in states.species:
                raise ValueError(f"dose target has unknown species: {species_id}")
            if states.spec(state_name).species_id != species_id:
                raise ValueError(
                    f"dose state {state_name} does not carry species {species_id}"
                )
        for species_id, state_name in {
            **central_states,
            **urine_states,
        }.items():
            if states.spec(state_name).species_id != species_id:
                raise ValueError(
                    f"observation state {state_name} does not carry species {species_id}"
                )
        factors = {str(key): float(value) for key, value in (mechanism_factors or {}).items()}
        if any(
            not key.strip() or not np.isfinite(value) or value < 0.0
            for key, value in factors.items()
        ):
            raise ValueError("mechanism factors must be finite values >= 0")
        self.model_id = model_id
        self.drug = drug
        self.formulation = formulation
        self.state_registry = states
        self.assembler = assembler
        self.reaction_network = reaction_network
        self.ledger = ledger
        self.parent_species_id = parent_species_id
        self.dose_targets = MappingProxyType(dict(dose_targets))
        self.central_states = MappingProxyType(dict(central_states))
        self.urine_states = MappingProxyType(dict(urine_states))
        self.feces_parent_state = feces_parent_state
        self.parent_liver_states = tuple(parent_liver_states)
        self.mechanism_factors = MappingProxyType(factors)

    def _validate_drug(self, drug: DrugParameters) -> None:
        if drug != self.drug:
            raise ValueError(
                "multi-species model must be rebuilt when drug parameters change"
            )

    def initial_state(self) -> np.ndarray:
        return self.state_registry.zeros()

    def apply_species_doses(
        self,
        state: np.ndarray,
        events: Sequence[SpeciesDoseEvent],
    ) -> np.ndarray:
        values = self.state_registry.validate_vector(state).copy()
        for event in events:
            key = (event.route, event.species_id)
            try:
                target = self.dose_targets[key]
            except KeyError as error:
                raise ValueError(f"no registered dose target for {key}") from error
            species = self.state_registry.species[event.species_id]
            amount_umol = species.convert_amount(
                event.amount, event.unit, AmountUnit.UMOL
            )
            values[self.state_registry.index(target)] += float(amount_umol)
        return values

    def apply_doses(
        self,
        state: np.ndarray,
        events: Sequence[DoseEvent],
    ) -> np.ndarray:
        """Bridge legacy active-parent mg dose events into the µmol model."""

        converted = tuple(
            SpeciesDoseEvent(
                time_h=event.time_h,
                species_id=self.parent_species_id,
                amount=event.amount_mg,
                unit=AmountUnit.MG,
                route=event.route,
                dose_basis_id="legacy_active_parent_mg",
            )
            for event in events
        )
        return self.apply_species_doses(state, converted)

    def rhs(
        self,
        time_h: float,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> np.ndarray:
        self._validate_drug(drug)
        return self.assembler.rhs(
            time_h,
            state,
            AmountModelContext(
                patient=patient,
                drug=drug,
                formulation=self.formulation,
                mechanism_factors=self.mechanism_factors,
            ),
        )

    def _amount_umol(self, vector: np.ndarray, state_name: str) -> float:
        return max(
            float(vector[self.state_registry.index(state_name)]),
            0.0,
        )

    def _species_mass_mg(self, species_id: str, amount_umol: float) -> float:
        return float(
            self.state_registry.species[species_id].convert_amount(
                amount_umol, AmountUnit.UMOL, AmountUnit.MG
            )
        )

    def concentration_record(
        self,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> dict[str, float]:
        self._validate_drug(drug)
        values = self.state_registry.validate_vector(state)
        ledger = self.ledger.report(values)
        parent_central_umol = self._amount_umol(
            values, self.central_states[self.parent_species_id]
        )
        parent_central_mg = self._species_mass_mg(
            self.parent_species_id, parent_central_umol
        )

        non_parent_species = tuple(
            species_id
            for species_id in self.central_states
            if species_id != self.parent_species_id
        )
        metabolite_plasma_mg_l = sum(
            self._species_mass_mg(
                species_id,
                self._amount_umol(values, self.central_states[species_id]),
            )
            / patient.central_volume_l
            for species_id in non_parent_species
        )
        urine_parent_mg = self._species_mass_mg(
            self.parent_species_id,
            self._amount_umol(values, self.urine_states[self.parent_species_id]),
        )
        urine_metabolite_mg = sum(
            self._species_mass_mg(
                species_id,
                self._amount_umol(values, state_name),
            )
            for species_id, state_name in self.urine_states.items()
            if species_id != self.parent_species_id
        )
        feces_parent_mg = self._species_mass_mg(
            self.parent_species_id,
            self._amount_umol(values, self.feces_parent_state),
        )

        record: dict[str, float] = {
            "plasma_parent_mg_l": parent_central_mg / patient.central_volume_l,
            "plasma_metabolite_mg_l": float(metabolite_plasma_mg_l),
            "urine_parent_mg": urine_parent_mg,
            "urine_metabolite_mg": float(urine_metabolite_mg),
            "feces_parent_mg": feces_parent_mg,
            "other_products_mg": 0.0,
            # The historical simulator subtracts delivered parent mg from this
            # field. Parent-equivalent moiety mass is the correct invariant.
            "tracked_mass_mg": ledger.drug_equivalent_mass_mg,
            "drug_moiety_total_umol": ledger.moiety_umol.get(
                self.ledger.conserved_moiety_id, 0.0
            ),
            "drug_equivalent_mass_mg": ledger.drug_equivalent_mass_mg,
            "tracked_molecular_mass_mg": ledger.tracked_molecular_mass_mg,
            "external_import_mass_mg": ledger.external_import_mass_mg,
            "external_export_mass_mg": ledger.external_export_mass_mg,
            "augmented_accounted_mass_mg": ledger.augmented_accounted_mass_mg,
            "augmented_minus_moiety_mass_mg": (
                ledger.augmented_accounted_mass_mg
                - ledger.drug_equivalent_mass_mg
            ),
            "reaction_static_max_abs_closure_mg_per_umol": max(
                abs(item.closure_error_mg_per_umol)
                for item in self.reaction_network.balance_diagnostics
            ),
        }
        zone_volume_l = patient.liver_volume_l / 3.0
        for zone_index, state_name in enumerate(self.parent_liver_states, start=1):
            amount_mg = self._species_mass_mg(
                self.parent_species_id,
                self._amount_umol(values, state_name),
            )
            record[f"liver_zone{zone_index}_parent_mg_l"] = (
                amount_mg / zone_volume_l
            )

        for species_id, state_name in self.central_states.items():
            central_umol = self._amount_umol(values, state_name)
            record[f"plasma_{species_id}_umol_l"] = (
                central_umol / patient.central_volume_l
            )
            record[f"plasma_{species_id}_mg_l"] = (
                self._species_mass_mg(species_id, central_umol)
                / patient.central_volume_l
            )
        for species_id, state_name in self.urine_states.items():
            urine_umol = self._amount_umol(values, state_name)
            record[f"urine_{species_id}_umol"] = urine_umol
            record[f"urine_{species_id}_mg"] = self._species_mass_mg(
                species_id, urine_umol
            )
        for species_id in self.central_states:
            total_umol = sum(
                self._amount_umol(values, state_name)
                for state_name in self.state_registry.names_for_species(species_id)
            )
            record[f"total_{species_id}_umol"] = total_umol
        return record
