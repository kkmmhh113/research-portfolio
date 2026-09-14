"""Post-v0.3 population specifications and reproducible virtual subjects.

This namespace is additive.  It does not replace or mutate the frozen
``our_star.virtual_population`` v0.3 implementation.
"""

from .random_effects import LatentDraws, RandomEffectSpec, draw_latent_effects
from .sampling import (
    PopulationSample,
    PopulationSamplingTrace,
    sample_virtual_subjects,
    virtual_subject_records,
)
from .specification import (
    FlowCompositionSpec,
    ModifierDomain,
    ModifierSpec,
    ModifierTransform,
    PopulationSpecification,
    VirtualSubject,
    apply_latent_transform,
)

__all__ = [
    "FlowCompositionSpec",
    "LatentDraws",
    "ModifierDomain",
    "ModifierSpec",
    "ModifierTransform",
    "PopulationSample",
    "PopulationSamplingTrace",
    "PopulationSpecification",
    "RandomEffectSpec",
    "VirtualSubject",
    "apply_latent_transform",
    "draw_latent_effects",
    "sample_virtual_subjects",
    "virtual_subject_records",
]
