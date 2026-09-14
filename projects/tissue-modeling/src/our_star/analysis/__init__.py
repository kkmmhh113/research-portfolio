"""Validated analysis utilities for research-only PBPK models.

The package deliberately accepts plain evaluator callables instead of importing a
particular PBPK implementation.  This keeps sensitivity and identifiability
analysis reusable across organ assemblies and prevents analysis code from silently
changing model or clinical-data state.
"""

from .identifiability import (
    IdentifiabilityDiagnostics,
    JacobianResult,
    finite_difference_jacobian,
    identifiability_diagnostics,
)
from .parameters import ParameterSpace, ParameterSpec
from .profiling import ProfileLikelihoodResult, profile_likelihood
from .sampling import SamplingDesign, sample_parameter_space
from .sensitivity import SensitivityResult, partial_rank_correlation
from .uncertainty import (
    CoverageResult,
    EmpiricalIntervals,
    empirical_intervals,
    observed_coverage,
)

__all__ = [
    "CoverageResult",
    "EmpiricalIntervals",
    "IdentifiabilityDiagnostics",
    "JacobianResult",
    "ParameterSpace",
    "ParameterSpec",
    "ProfileLikelihoodResult",
    "SamplingDesign",
    "SensitivityResult",
    "empirical_intervals",
    "finite_difference_jacobian",
    "identifiability_diagnostics",
    "observed_coverage",
    "partial_rank_correlation",
    "profile_likelihood",
    "sample_parameter_space",
]
