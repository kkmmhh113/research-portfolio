"""Adapters from v0.4 latent subjects to the frozen PBPK physiology record."""

from __future__ import annotations

from dataclasses import fields, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..pbpk import PatientPhysiology
from .specification import VirtualSubject


_FLOW_FIELDS = (
    "portal_flow_l_h",
    "hepatic_artery_flow_l_h",
    "renal_flow_l_h",
    "rest_flow_l_h",
)


def materialize_patient_physiology(
    subject: VirtualSubject,
    reference: PatientPhysiology,
) -> PatientPhysiology:
    """Apply absolute anatomy values while keeping mechanisms separate.

    ``VirtualSubject.anatomy_modifiers`` are physical values after their
    constrained transforms, not multipliers.  The optional
    ``cardiac_output_l_h`` target is an audit value: the four component flows
    must also be supplied and sum to it before construction proceeds.
    """

    if not isinstance(subject, VirtualSubject):
        raise TypeError("subject must be a v0.4 VirtualSubject")
    if not isinstance(reference, PatientPhysiology):
        raise TypeError("reference must be PatientPhysiology")
    anatomy = dict(subject.anatomy_modifiers)
    declared_cardiac_output = anatomy.pop("cardiac_output_l_h", None)
    allowed = {
        item.name
        for item in fields(PatientPhysiology)
        if item.name not in {"patient_id", "sex", "genotype_label"}
    }
    unknown = sorted(set(anatomy) - allowed)
    if unknown:
        raise ValueError(f"anatomy modifiers target unsupported fields: {unknown}")
    if declared_cardiac_output is not None:
        missing_flows = sorted(set(_FLOW_FIELDS) - set(anatomy))
        if missing_flows:
            raise ValueError(
                "cardiac_output_l_h requires all component flows: "
                f"{missing_flows}"
            )
        realized = sum(float(anatomy[name]) for name in _FLOW_FIELDS)
        tolerance = 8.0 * float(np.spacing(float(declared_cardiac_output)))
        if not np.isclose(
            realized,
            float(declared_cardiac_output),
            rtol=0.0,
            atol=max(tolerance, 1.0e-12),
        ):
            raise ValueError("component flows do not match declared cardiac output")
    return replace(reference, patient_id=subject.subject_id, **anatomy)


def mechanism_factors(subject: VirtualSubject) -> Mapping[str, float]:
    """Return an immutable copy suitable for ``AmountModelContext``."""

    if not isinstance(subject, VirtualSubject):
        raise TypeError("subject must be a v0.4 VirtualSubject")
    factors = {name: float(value) for name, value in subject.mechanism_modifiers.items()}
    invalid = {
        name: value
        for name, value in factors.items()
        if not name.strip() or not np.isfinite(value) or value < 0.0
    }
    if invalid:
        raise ValueError(f"invalid mechanism factors: {invalid}")
    return MappingProxyType(factors)
