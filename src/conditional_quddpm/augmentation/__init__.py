"""Train-only state augmentation utilities."""

from .physics import augment_state, fit_acceptance_gate, tfim_components

__all__ = ["augment_state", "fit_acceptance_gate", "tfim_components"]
