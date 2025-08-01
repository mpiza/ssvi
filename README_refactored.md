# SSVI Volatility Models - Refactored

This project provides implementations of SVI (Stochastic Volatility Inspired) and SSVI (Surface SVI) volatility models for derivatives pricing and risk management. The code has been refactored to separate core calculations from visualization and application code.

## Project Structure

```
ssvi/
├── README.md                     # This file
├── __init__.py                   # Package initialization
├── svi_models.py                 # Core SVI and SSVI model calculations
├── density_analysis.py           # Risk-neutral density analysis functions
├── volatility_surface_app.py     # Interactive visualization application
├── examples.py                   # Example usage of core modules
└── tests/                        # Test files and investigations
    ├── compare_density_formulas.py
    ├── test_extreme_svi.py
    ├── verify_density.py
    └── ...
```

## Refactoring Overview

The original monolithic `volatility_surface_app.py` has been refactored into:

### 1. Core Calculation Modules

#### `svi_models.py`
Contains the fundamental SVI and SSVI model calculations:
- `compute_svi_total_variance()` - SVI total variance computation
- `compute_svi_volatility_smile()` - Single maturity SVI smile
- `compute_svi_surface()` - Full SVI volatility surface
- `compute_ssvi_total_variance()` - SSVI total variance computation
- `compute_ssvi_surface()` - Full SSVI volatility surface
- `validate_svi_parameters()` - Parameter validation for SVI
- `validate_ssvi_parameters()` - Parameter validation for SSVI

#### `density_analysis.py`
Contains risk-neutral density analysis functions:
- `compute_svi_risk_neutral_density()` - Breeden-Litzenberger density calculation
- `verify_density_properties()` - Mathematical property verification
- `analyze_density_arbitrage()` - Arbitrage detection and analysis
- `compute_moment_from_density()` - Distribution moment calculation

### 2. Application Layer

#### `volatility_surface_app.py` (Updated)
Interactive visualization application that now imports and uses the core calculation modules:
- SVI 3D surface visualization
- SSVI 3D surface visualization  
- SVI smile with density analysis
- Interactive parameter adjustment with sliders

#### `examples.py` (New)
Demonstrates how to use the separated core modules independently:
- Basic SVI smile computation and plotting
- Risk-neutral density analysis with arbitrage detection
- SVI vs SSVI surface comparison

## Usage

### Running the Interactive Application

```bash
python volatility_surface_app.py
```

Choose from:
1. SVI 3D surface visualization
2. SSVI 3D surface visualization
3. SVI smile with risk-neutral density analysis

### Using Core Modules Independently

```python
from svi_models import compute_svi_volatility_smile, validate_svi_parameters
from density_analysis import compute_svi_risk_neutral_density, verify_density_properties

import numpy as np

# Define parameters
k_values = np.linspace(-2.0, 2.0, 100)
T = 0.5
a, b, rho, m, sigma = 0.02, 0.4, -0.2, 0.0, 0.4

# Validate parameters
is_valid, violations = validate_svi_parameters(a, b, rho, m, sigma)
print(f"Parameters valid: {is_valid}")

# Compute volatility smile
vol_smile = compute_svi_volatility_smile(k_values, T, a, b, rho, m, sigma)

# Compute and analyze risk-neutral density
density = compute_svi_risk_neutral_density(k_values, T, a, b, rho, m, sigma)
verification = verify_density_properties(k_values, density)
```

### Running Examples

```bash
python examples.py
```

This will run demonstration examples showing:
- SVI volatility smile computation and plotting
- Risk-neutral density analysis with arbitrage detection
- Comparison between SVI and SSVI surfaces

## Key Benefits of Refactoring

1. **Separation of Concerns**: Core calculations are now independent of visualization code
2. **Reusability**: SVI model functions can be easily imported and used in other projects
3. **Testability**: Individual functions can be unit tested independently
4. **Maintainability**: Changes to calculations don't affect UI code and vice versa
5. **Extensibility**: New analysis functions can be added without modifying existing code
6. **Documentation**: Each module has a clear, focused purpose with comprehensive docstrings

## Mathematical Models

### SVI Model
The SVI (Stochastic Volatility Inspired) model parameterizes total variance as:

```
w(k) = a + b * (ρ(k - m) + √((k - m)² + σ²))
```

Where:
- `a`: level (intercept)
- `b`: angle (slope parameter)
- `ρ`: correlation parameter (-1 < ρ < 1)
- `m`: horizontal shift
- `σ`: curvature parameter (σ > 0)

### SSVI Model  
The SSVI (Surface SVI) model parameterizes total variance as:

```
w(k) = 0.5 * θ * (1 + ρφk + √((φk + ρ)² + 1 - ρ²))
```

Where:
- `θ`: variance level (θ > 0)
- `φ`: skew parameter (φ > 0)  
- `ρ`: correlation parameter (-1 < ρ < 1)

### Risk-Neutral Density
The risk-neutral density is computed using the Breeden-Litzenberger formula with the complete shape function that accounts for the curvature and derivatives of the SVI total variance function.

## Dependencies

- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `matplotlib.widgets`: Interactive sliders and buttons

## Future Enhancements

The modular structure makes it easy to add:
- Additional volatility models (Heston, SABR, etc.)
- More sophisticated arbitrage detection
- Parameter calibration functions
- Monte Carlo simulation capabilities
- Advanced plotting and analysis tools
