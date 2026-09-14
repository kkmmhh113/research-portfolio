"""Amount-conserving gut, liver, kidney, and rest-of-body modules."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..core import ModelContext, ModuleResult, StateRegistry
from ..pbpk import _michaelis_menten


def _amount(states: StateRegistry, amounts_mg: np.ndarray, name: str) -> float:
    return float(amounts_mg[states.index(name)])


class LegacyGutModule:
    """Original one-lumen gut, expressed through the organ-module contract."""

    name = "legacy_gut"
    required_states = (
        "gut_lumen_parent",
        "gut_wall_parent",
        "central_parent",
        "central_metabolite",
        "feces_parent",
        "other_products",
    )
    required_inputs: tuple[str, ...] = ()

    def evaluate(
        self,
        _time_h: float,
        amounts_mg: np.ndarray,
        context: ModelContext,
        states: StateRegistry,
        _inputs_mg_h: Mapping[str, float],
    ) -> ModuleResult:
        patient = context.patient
        drug = context.drug
        central = _amount(states, amounts_mg, "central_parent") / patient.central_volume_l
        gut_equivalent = (
            _amount(states, amounts_mg, "gut_wall_parent")
            / patient.gut_volume_l
            / drug.kp_gut
        )
        lumen_loss = drug.absorption_rate_h * _amount(
            states, amounts_mg, "gut_lumen_parent"
        )
        absorbed = drug.fraction_absorbed * lumen_loss
        fecal = (1.0 - drug.fraction_absorbed) * lumen_loss
        gut_vmax = (
            drug.gut_vmax_mg_h
            * patient.enzyme_activity_factor
            * patient.gut_volume_l
            / 1.2
        )
        gut_metabolism = _michaelis_menten(
            gut_vmax,
            drug.gut_km_mg_l,
            drug.fraction_unbound_plasma * gut_equivalent,
        )
        formed_metabolite = drug.metabolite_mass_yield * gut_metabolism
        other_products = (1.0 - drug.metabolite_mass_yield) * gut_metabolism
        portal_return = patient.portal_flow_l_h * gut_equivalent
        derivatives = {
            "gut_lumen_parent": -lumen_loss,
            "feces_parent": fecal,
            "gut_wall_parent": (
                absorbed
                + patient.portal_flow_l_h * central
                - portal_return
                - gut_metabolism
            ),
            "central_parent": -patient.portal_flow_l_h * central,
            "central_metabolite": formed_metabolite,
            "other_products": other_products,
        }
        return ModuleResult(
            derivatives_mg_h=derivatives,
            outputs_mg_h={
                "portal_parent_mg_h": portal_return,
                "gut_metabolism_mg_h": gut_metabolism,
            },
        )


class SegmentedGutModule:
    """Dissolution, gastric emptying, and three intestinal absorption segments."""

    name = "segmented_gi"
    segments = ("duodenum", "jejunum", "ileum")
    required_states = (
        "stomach_solid_parent",
        "stomach_dissolved_parent",
        "duodenum_lumen_parent",
        "duodenum_wall_parent",
        "jejunum_lumen_parent",
        "jejunum_wall_parent",
        "ileum_lumen_parent",
        "ileum_wall_parent",
        "central_parent",
        "central_metabolite",
        "feces_parent",
        "other_products",
    )
    required_inputs: tuple[str, ...] = ()

    def evaluate(
        self,
        _time_h: float,
        amounts_mg: np.ndarray,
        context: ModelContext,
        states: StateRegistry,
        _inputs_mg_h: Mapping[str, float],
    ) -> ModuleResult:
        patient = context.patient
        drug = context.drug
        formulation = context.formulation
        if formulation is None:
            raise ValueError("SegmentedGutModule requires formulation parameters")

        central = _amount(states, amounts_mg, "central_parent") / patient.central_volume_l
        solid = _amount(states, amounts_mg, "stomach_solid_parent")
        dissolved = _amount(states, amounts_mg, "stomach_dissolved_parent")
        dissolution = formulation.dissolution_rate_h * solid
        gastric_emptying = formulation.gastric_emptying_rate_h * dissolved
        derivatives: dict[str, float] = {
            "stomach_solid_parent": -dissolution,
            "stomach_dissolved_parent": dissolution - gastric_emptying,
        }

        segment_returns: list[float] = []
        segment_metabolism: list[float] = []
        incoming_lumen = gastric_emptying
        reference_gut_volume_l = 1.2
        gut_vmax_total = (
            drug.gut_vmax_mg_h
            * patient.enzyme_activity_factor
            * patient.gut_volume_l
            / reference_gut_volume_l
        )
        for index, segment in enumerate(self.segments):
            lumen_name = f"{segment}_lumen_parent"
            wall_name = f"{segment}_wall_parent"
            lumen = _amount(states, amounts_mg, lumen_name)
            wall = _amount(states, amounts_mg, wall_name)
            absorption = (
                drug.fraction_absorbed
                * formulation.segment_absorption_rate_h[index]
                * lumen
            )
            transit = formulation.segment_transit_rate_h[index] * lumen
            wall_volume = patient.gut_volume_l * formulation.wall_volume_fraction[index]
            wall_equivalent = wall / wall_volume / drug.kp_gut
            segment_flow = patient.portal_flow_l_h * formulation.portal_flow_fraction[index]
            venous_return = segment_flow * wall_equivalent
            metabolism = _michaelis_menten(
                gut_vmax_total * formulation.wall_volume_fraction[index],
                drug.gut_km_mg_l,
                drug.fraction_unbound_plasma * wall_equivalent,
            )
            derivatives[lumen_name] = incoming_lumen - absorption - transit
            derivatives[wall_name] = (
                absorption
                + segment_flow * central
                - venous_return
                - metabolism
            )
            segment_returns.append(venous_return)
            segment_metabolism.append(metabolism)
            incoming_lumen = transit

        total_return = float(sum(segment_returns))
        total_metabolism = float(sum(segment_metabolism))
        derivatives["feces_parent"] = incoming_lumen
        derivatives["central_parent"] = -patient.portal_flow_l_h * central
        derivatives["central_metabolite"] = (
            drug.metabolite_mass_yield * total_metabolism
        )
        derivatives["other_products"] = (
            (1.0 - drug.metabolite_mass_yield) * total_metabolism
        )
        outputs = {
            "portal_parent_mg_h": total_return,
            "gut_metabolism_mg_h": total_metabolism,
            "gastric_emptying_mg_h": gastric_emptying,
            **{
                f"{segment}_absorption_mg_h": (
                    drug.fraction_absorbed
                    * formulation.segment_absorption_rate_h[index]
                    * _amount(states, amounts_mg, f"{segment}_lumen_parent")
                )
                for index, segment in enumerate(self.segments)
            },
        }
        return ModuleResult(derivatives_mg_h=derivatives, outputs_mg_h=outputs)


class LiverModule:
    """Three serial perfused liver zones with saturable parent metabolism."""

    name = "liver"
    required_states = (
        "central_parent",
        "central_metabolite",
        "liver_zone1_parent",
        "liver_zone2_parent",
        "liver_zone3_parent",
        "other_products",
    )
    required_inputs = ("portal_parent_mg_h",)

    def evaluate(
        self,
        _time_h: float,
        amounts_mg: np.ndarray,
        context: ModelContext,
        states: StateRegistry,
        inputs_mg_h: Mapping[str, float],
    ) -> ModuleResult:
        patient = context.patient
        drug = context.drug
        central = _amount(states, amounts_mg, "central_parent") / patient.central_volume_l
        zone_volume = patient.liver_volume_l / 3.0
        liver_equivalent = np.asarray(
            [
                _amount(states, amounts_mg, f"liver_zone{zone}_parent")
                / zone_volume
                / drug.kp_liver
                for zone in (1, 2, 3)
            ],
            dtype=np.float64,
        )
        liver_scale = (
            patient.enzyme_activity_factor
            * patient.liver_function_fraction
            * patient.liver_volume_l
            / 1.8
        )
        liver_metabolism = np.asarray(
            [
                _michaelis_menten(
                    drug.liver_vmax_mg_h
                    * liver_scale
                    * drug.normalized_zone_activity[index],
                    drug.liver_km_mg_l,
                    drug.fraction_unbound_plasma * liver_equivalent[index],
                )
                for index in range(3)
            ],
            dtype=np.float64,
        )
        q_liver = patient.liver_flow_l_h
        derivatives = {
            "liver_zone1_parent": (
                inputs_mg_h["portal_parent_mg_h"]
                + patient.hepatic_artery_flow_l_h * central
                - q_liver * liver_equivalent[0]
                - liver_metabolism[0]
            ),
            "liver_zone2_parent": (
                q_liver * liver_equivalent[0]
                - q_liver * liver_equivalent[1]
                - liver_metabolism[1]
            ),
            "liver_zone3_parent": (
                q_liver * liver_equivalent[1]
                - q_liver * liver_equivalent[2]
                - liver_metabolism[2]
            ),
            "central_parent": (
                q_liver * liver_equivalent[2]
                - patient.hepatic_artery_flow_l_h * central
            ),
            "central_metabolite": (
                drug.metabolite_mass_yield * float(liver_metabolism.sum())
            ),
            "other_products": (
                (1.0 - drug.metabolite_mass_yield) * float(liver_metabolism.sum())
            ),
        }
        return ModuleResult(
            derivatives_mg_h=derivatives,
            outputs_mg_h={
                "hepatic_venous_parent_mg_h": q_liver * liver_equivalent[2],
                "liver_metabolism_mg_h": float(liver_metabolism.sum()),
            },
        )


class KidneyModule:
    """Perfusion-limited kidney and renal parent/metabolite elimination."""

    name = "kidney"
    required_states = (
        "central_parent",
        "kidney_parent",
        "central_metabolite",
        "urine_parent",
        "urine_metabolite",
    )
    required_inputs: tuple[str, ...] = ()

    def evaluate(
        self,
        _time_h: float,
        amounts_mg: np.ndarray,
        context: ModelContext,
        states: StateRegistry,
        _inputs_mg_h: Mapping[str, float],
    ) -> ModuleResult:
        patient = context.patient
        drug = context.drug
        central = _amount(states, amounts_mg, "central_parent") / patient.central_volume_l
        kidney_equivalent = (
            _amount(states, amounts_mg, "kidney_parent")
            / patient.kidney_volume_l
            / drug.kp_kidney
        )
        renal_scale = (
            patient.renal_function_fraction * (patient.body_weight_kg / 70.0) ** 0.75
        )
        parent_elimination = (
            drug.renal_clearance_l_h * renal_scale * kidney_equivalent
        )
        metabolite_concentration = (
            _amount(states, amounts_mg, "central_metabolite")
            / patient.central_volume_l
        )
        metabolite_elimination = (
            drug.metabolite_renal_clearance_l_h
            * renal_scale
            * metabolite_concentration
        )
        exchange = patient.renal_flow_l_h * (central - kidney_equivalent)
        return ModuleResult(
            derivatives_mg_h={
                "kidney_parent": exchange - parent_elimination,
                "central_parent": -exchange,
                "urine_parent": parent_elimination,
                "central_metabolite": -metabolite_elimination,
                "urine_metabolite": metabolite_elimination,
            },
            outputs_mg_h={
                "renal_parent_elimination_mg_h": parent_elimination,
                "renal_metabolite_elimination_mg_h": metabolite_elimination,
            },
        )


class RestOfBodyModule:
    """Perfusion-limited aggregate of tissues not yet modeled explicitly."""

    name = "rest_of_body"
    required_states = ("central_parent", "rest_parent")
    required_inputs: tuple[str, ...] = ()

    def evaluate(
        self,
        _time_h: float,
        amounts_mg: np.ndarray,
        context: ModelContext,
        states: StateRegistry,
        _inputs_mg_h: Mapping[str, float],
    ) -> ModuleResult:
        patient = context.patient
        drug = context.drug
        central = _amount(states, amounts_mg, "central_parent") / patient.central_volume_l
        rest_equivalent = (
            _amount(states, amounts_mg, "rest_parent")
            / patient.rest_volume_l
            / drug.kp_rest
        )
        exchange = patient.rest_flow_l_h * (central - rest_equivalent)
        return ModuleResult(
            derivatives_mg_h={
                "rest_parent": exchange,
                "central_parent": -exchange,
            }
        )
