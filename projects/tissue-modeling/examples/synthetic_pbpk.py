"""Run a synthetic PBPK example; its parameters are not clinical estimates."""

import json
from pathlib import Path

from our_star.models import SegmentedPBPKModel
from our_star.pbpk import DoseEvent
from our_star.virtual_population import reference_patient
from our_star.virtual_trial import load_drug, load_formulation, simulate_patient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    drug = load_drug(root / "configs/virtual_phase1/synthetic_probe.yaml")
    formulation = load_formulation(root / "configs/formulations/synthetic_probe_ir.yaml")
    result = simulate_patient(
        reference_patient(), drug, (DoseEvent(0.0, 100.0),),
        duration_h=12.0, output_interval_h=0.25,
        model=SegmentedPBPKModel(drug, formulation),
    )
    print(json.dumps({
        "example": "synthetic solver demonstration",
        "model": result.model_id,
        "time_points": len(result.times_h),
        "max_mass_balance_error_mg": result.summary["max_abs_mass_balance_error_mg"],
        "minimum_state_amount_mg": float(result.states_mg.min()),
    }, indent=2))


if __name__ == "__main__":
    main()
