# Project Organization Summary

This document describes the organized structure of the SSVI project after cleanup and reorganization.

## Main Directory Structure

```
ssvi/
├── README.md                               # Main project documentation
├── MATHEMATICAL_DERIVATIONS.md             # Complete mathematical documentation
├── PROJECT_ORGANIZATION.md                 # This organization guide
├── parametric_ssvi.py                      # Core parametric SSVI implementation
├── svi_models.py                           # Basic SVI models
├── local_volatility.py                     # Local volatility (Dupire formula)
├── density_analysis.py                     # Risk-neutral density analysis
├── apps/                                   # Interactive applications
│   ├── parametric_ssvi_iv_lv_density_app.py    # Main interactive app ⭐
│   ├── parametric_ssvi_app.py                  # Basic parametric SSVI app
│   ├── enhanced_parametric_ssvi_app.py         # Enhanced version
│   ├── simple_parametric_ssvi_app.py           # Simplified version
│   └── volatility_surface_app.py               # General volatility tool
├── analysis/                               # Analysis and benchmarking tools
│   ├── performance_analysis.py                 # Performance benchmarking
│   ├── quick_iv_lv_comparison.py              # IV/LV comparison tool
│   └── test_derivatives.py                     # Derivative validation
├── tests/                                  # All test scripts and verification
│   ├── test_analytical_derivatives.py          # Analytical derivatives tests
│   ├── test_phi_derivative_accuracy.py         # φ derivative accuracy
│   ├── run_analytical_tests.py                 # Test runner
│   └── plots/                                  # Generated test plots
├── examples/                               # Example usage scripts
│   ├── basic_examples.py                       # Basic usage examples
│   ├── local_volatility_examples.py            # Local volatility examples
│   └── parametric_ssvi_examples.py             # Parametric SSVI examples
├── archive/                                # Legacy and historical files
└── docs/                                   # Additional documentation
```

## Core Implementation Files

### `parametric_ssvi.py`
**Purpose**: Complete parametric SSVI implementation with analytical derivatives
**Key Features**:
- Time-dependent variance level θ(T) with mean reversion
- Rational function φ(θ) with analytical derivative ∂φ/∂θ
- All analytical derivatives: ∂w/∂μ, ∂²w/∂μ², ∂w/∂T
- Parameter validation and edge case handling

### `parametric_ssvi_iv_lv_density_app.py`
**Purpose**: Interactive application for IV, LV, and density visualization
**Key Features**:
- Real-time parameter adjustment with sliders
- Line plots for IV with consistent viridis colormap
- Line plots for LV (improved from contour surface)
- Density surface plots
- Centered two-column slider layout

### `local_volatility.py`
**Purpose**: Local volatility computation using Dupire formula
**Key Features**:
- Dupire formula implementation for total variance
- Numerical stability safeguards
- Support for both SVI and parametric SSVI models

## Test Organization

### `tests/` Directory
All test scripts have been moved to the `tests/` directory with proper organization:

#### Analytical Derivative Tests
- `test_analytical_derivatives.py`: Complete parametric SSVI derivatives validation
- `test_phi_derivative_accuracy.py`: φ(θ) derivative accuracy verification
- `run_analytical_tests.py`: Test runner for all analytical derivative tests

#### SVI Density Tests
- `verify_density.py`: Risk-neutral density verification suite
- `compare_density_formulas.py`: Old vs new density formula comparison
- `test_extreme_svi.py`: Extreme parameter testing

#### Local Volatility Tests
- `test_constant_volatility.py`: Dupire formula validation
- `test_realistic_local_vol.py`: Realistic market parameter testing
- `test_local_vol.py`: General local volatility tests

#### Plots Organization
- `tests/plots/`: All generated plots are stored here
  - `parametric_ssvi_derivatives_test.png`
  - `phi_derivative_verification.png`
  - `derivative_comparison.png`
  - `parametric_ssvi_performance_analysis.png`
  - Plus existing density verification plots

## Applications

### Interactive Applications (`apps/`)
- `parametric_ssvi_iv_lv_density_app.py`: **Main interactive tool** ⭐
  - Complete IV/LV/density visualization
  - Real-time parameter adjustment
  - Analytical derivatives for performance
  - Centered slider layout with consistent viridis colormap
- `parametric_ssvi_app.py`: Basic parametric SSVI application
- `enhanced_parametric_ssvi_app.py`: Extended version with additional features
- `simple_parametric_ssvi_app.py`: Simplified educational version
- `volatility_surface_app.py`: General volatility surface tool

### Analysis Tools (`analysis/`)
- `performance_analysis.py`: Performance benchmarking and optimization
- `quick_iv_lv_comparison.py`: Quick IV/LV comparison and validation
- `test_derivatives.py`: Comprehensive derivative testing and validation

## Removed Files

### Temporary Test Files (Cleaned Up)
- `simple_test.py`: Basic verification (functionality moved to organized tests)
- Root-level plot files: Moved to `tests/plots/`

### Redundant or Outdated
- Old temporary scripts created during development

## Key Improvements Made

### 1. Analytical Derivatives Implementation ✅
- Replaced numerical φ derivative with exact analytical formula
- All derivatives now computed with machine precision
- Consistent with mathematical documentation

### 2. Test Organization ✅
- Clear separation of test types
- Proper import paths for tests
- Centralized plot storage
- Comprehensive test runner

### 3. Documentation Consistency ✅
- Mathematical documentation matches implementation
- Clear project structure documentation
- Updated test documentation

### 4. Application Polish ✅
- Improved layout with centered sliders
- Consistent colormap across all plots
- Better visual presentation

## Usage Instructions

### Running Interactive Applications
```bash
# Main interactive application (recommended)
python apps/parametric_ssvi_iv_lv_density_app.py

# Other applications
python apps/parametric_ssvi_app.py
python apps/enhanced_parametric_ssvi_app.py
python apps/simple_parametric_ssvi_app.py
```

### Running Analysis Tools
```bash
# Performance benchmarking
python analysis/performance_analysis.py

# Quick IV/LV comparison
python analysis/quick_iv_lv_comparison.py

# Derivative validation
python analysis/test_derivatives.py
```

### Running Tests
```bash
# All analytical derivative tests
python tests/run_analytical_tests.py

# Individual tests
python tests/test_analytical_derivatives.py
python tests/test_phi_derivative_accuracy.py

# SVI density verification
python tests/verify_density.py
```

### Using Core Modules
```python
# Import core functionality
from parametric_ssvi import (
    compute_parametric_ssvi_all_derivatives,
    compute_parametric_ssvi_total_variance
)

# Import local volatility
from local_volatility import dupire_local_volatility_from_total_variance

# Import density analysis
from density_analysis import compute_svi_density
```

## Mathematical Foundation

The complete mathematical foundation is documented in `MATHEMATICAL_DERIVATIONS.md`, which now accurately reflects the implementation:

1. **Parametric SSVI Model**: w(μ, T) with time-dependent parameters
2. **Analytical Derivatives**: All derivatives computed exactly
3. **Dupire Formula**: Local volatility with numerical stability
4. **Implementation Notes**: Practical considerations and edge cases

This organization provides a clean, maintainable, and well-documented parametric SSVI implementation with comprehensive testing and verification capabilities.
