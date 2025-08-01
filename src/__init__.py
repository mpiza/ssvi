"""
SSVI Core Modules

This package contains the core implementation modules for the SSVI project:
- parametric_ssvi.py: Main parametric SSVI implementation with analytical derivatives
- svi_models.py: SVI model implementations and utilities
- local_volatility.py: Local volatility calculation utilities
- density_analysis.py: Risk-neutral density analysis tools

All modules provide comprehensive implementations with proper error handling
and numerical stability considerations.
"""

# Core SSVI implementation
from .parametric_ssvi import *
from .svi_models import *
from .local_volatility import *
from .density_analysis import *

__version__ = "1.0.0"
__author__ = "SSVI Project"
__all__ = [
    # Parametric SSVI functions
    "compute_parametric_ssvi_total_variance",
    "compute_parametric_ssvi_surface", 
    "compute_parametric_ssvi_local_volatility",
    "compute_parametric_ssvi_derivatives",
    "validate_parametric_ssvi_parameters",
    
    # SVI model functions
    "compute_svi_total_variance",
    "compute_svi_volatility_smile",
    "compute_ssvi_total_variance",
    "validate_svi_parameters",
    "validate_ssvi_parameters",
    
    # Local volatility functions
    "compute_svi_local_volatility",
    "compute_ssvi_local_volatility",
    "dupire_local_volatility_from_total_variance",
    
    # Density analysis functions
    "compute_svi_risk_neutral_density",
    "verify_density_properties",
    "analyze_density_arbitrage"
]
