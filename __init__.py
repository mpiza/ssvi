"""
SSVI (Stochastic Volatility Inspired and Surface SVI) Models Package

This package provides implementations of SVI and SSVI volatility models
for derivatives pricing and risk management.
"""

from svi_models import (
    compute_svi_surface,
    compute_ssvi_surface,
    compute_svi_volatility_smile,
    compute_svi_total_variance,
    compute_ssvi_total_variance,
    validate_svi_parameters,
    validate_ssvi_parameters
)

from density_analysis import (
    compute_svi_risk_neutral_density,
    verify_density_properties,
    analyze_density_arbitrage,
    compute_moment_from_density
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # SVI model functions
    "compute_svi_surface",
    "compute_ssvi_surface", 
    "compute_svi_volatility_smile",
    "compute_svi_total_variance",
    "compute_ssvi_total_variance",
    "validate_svi_parameters",
    "validate_ssvi_parameters",
    # Density analysis functions
    "compute_svi_risk_neutral_density",
    "verify_density_properties",
    "analyze_density_arbitrage",
    "compute_moment_from_density"
]
