# SSVI Volatility Models - Refactored

This project provides implementations of SVI (Stochastic Volatility Inspired) and SSVI (Surface SVI) volatility models for derivatives pricing and risk management. The code has been refactored to separate core calculations from visualization and application code.

## Project Structure

```
ssvi/
├── README.md                     # This file
├── .gitignore                    # Git ignore file for Python projects
├── __init__.py                   # Package initialization
├── svi_models.py                 # Core SVI and SSVI model calculations
├── density_analysis.py           # Risk-neutral density analysis functions
├── local_volatility.py           # Local volatility computation via Dupire formula
├── volatility_surface_app.py     # Interactive visualization application
├── examples.py                   # Example usage of core modules
├── local_volatility_examples.py  # Local volatility examples and demonstrations
├── test_local_vol.py             # Basic tests for local volatility functionality
├── test_constant_volatility.py   # Test for constant volatility edge case
├── test_realistic_local_vol.py   # Realistic slice-by-slice calibration test
├── explain_time_derivatives.py   # Detailed explanation of time derivative handling
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

#### `local_volatility.py` (New)
Contains local volatility computation using the Dupire formula:
- `compute_svi_local_volatility()` - Local volatility from SVI parameters
- `compute_ssvi_local_volatility()` - Local volatility from SSVI parameters
- `dupire_local_volatility_from_total_variance()` - Core Dupire formula implementation
- `analyze_local_volatility_properties()` - Local volatility analysis and validation
- `compare_implied_vs_local_volatility()` - Comprehensive comparison framework

### 2. Application Layer

#### `volatility_surface_app.py` (Updated)
Interactive visualization application that now imports and uses the core calculation modules:
- SVI 3D surface visualization
- SSVI 3D surface visualization  
- SVI smile with density analysis
- Interactive parameter adjustment with sliders

#### `examples.py` (Updated)
Demonstrates how to use the separated core modules independently:
- Basic SVI smile computation and plotting
- Risk-neutral density analysis with arbitrage detection
- SVI vs SSVI surface comparison

#### `local_volatility_examples.py` (New)
Comprehensive examples for local volatility computation:
- SVI and SSVI local volatility computation and analysis
- Comparison between implied and local volatilities
- Sensitivity analysis of local volatility to model parameters
- Advanced plotting and visualization of local volatility properties

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
from local_volatility import compute_svi_local_volatility, analyze_local_volatility_properties

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

# Compute and analyze local volatility
local_vol, is_valid, diagnostics = compute_svi_local_volatility(
    k_values, T, a, b, rho, m, sigma, r=0.02
)
local_vol_analysis = analyze_local_volatility_properties(k_values, local_vol, is_valid)
```

### Running Examples

```bash
python examples.py                    # Basic SVI/SSVI examples
python local_volatility_examples.py  # Local volatility examples
python test_local_vol.py              # Quick local volatility tests
python test_constant_volatility.py   # Test constant volatility edge case
python test_realistic_local_vol.py   # Realistic slice-by-slice calibration
python explain_time_derivatives.py   # Detailed time derivative analysis
```

This will run demonstration examples showing:
- SVI volatility smile computation and plotting
- Risk-neutral density analysis with arbitrage detection
- Comparison between SVI and SSVI surfaces
- Local volatility computation and analysis
- Implied vs local volatility comparisons
- Realistic slice-by-slice calibration scenarios
- Calendar arbitrage detection and analysis
- Comprehensive time derivative analysis and comparison

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

### Local Volatility (New)
Local volatility is computed using the Dupire formula, which relates the local volatility function σ_LV(K,T) to the implied volatility surface and its derivatives:

```
σ_LV²(K,T) = (∂w/∂T + r·w) / (1 - k/(2w)·∂w/∂k + ¼·(-¼ - 1/w + k²/(4w²))·(∂w/∂k)² + ½·∂²w/∂k²)
```

Where w(k,T) = σ_IV²(K,T)·T is the total variance, and k = ln(K/S₀) is log-moneyness.

**Important Note on Time Derivatives**: In practice, SVI/SSVI models are calibrated slice-by-slice for each maturity, **not** with constant parameters across time. This means:

- ∂w/∂T must be computed from neighboring maturity slices using finite differences
- Constant SVI parameters across time can violate calendar arbitrage conditions  
- The assumption ∂w/∂T = 0 (time-homogeneous) is generally incorrect for market-calibrated surfaces

The implementation includes:
- Automatic derivative computation for SVI and SSVI models
- Realistic time derivative computation from neighboring slices
- Numerical stability checks and validation
- Arbitrage detection through negative local volatility identification
- Comprehensive analysis and comparison tools
- Calendar arbitrage checks for time-dependent surfaces

#### Detailed Time Derivative Handling

The time derivative ∂w/∂T is the most critical and challenging aspect of local volatility computation. Our implementation addresses both theoretical foundations and practical challenges:

**Three Approaches to Time Derivatives:**

1. **❌ Naive Approach**: ∂w/∂T = 0 (time-homogeneous assumption)
   - Problem: Incorrect for market-calibrated surfaces
   - Used in: Academic examples with constant parameters

2. **⚠️ Simplified Approach**: ∂w/∂T = w(k,T)/T  
   - Assumption: w(k,T) = T × w_svi(k) with constant SVI parameters
   - Problem: Only valid for unrealistic constant parameter surfaces
   - Result: Can differ significantly from market reality

3. **✅ Realistic Approach**: ∂w/∂T from finite differences between slices
   - Method: Compute (w₂ - w₁)/(T₂ - T₁) from neighboring maturity slices
   - Advantage: Captures actual market term structure effects
   - Implementation: `compute_time_derivative_from_slices()` function

**Practical Impact:**
- Time derivative differences can cause **30%+ variations** in local volatility levels
- Realistic slice-by-slice calibration often gives ∂w/∂T ≈ 0.5 × (w/T)
- Calendar arbitrage constraints require careful verification

**Code Examples:**
```python
# Simplified approach (often incorrect)
local_vol, is_valid = compute_svi_local_volatility(k_values, T, a, b, rho, m, sigma, r=0.02)

# Realistic approach (recommended)
dw_dT = compute_time_derivative_from_slices(k_values, maturities, svi_params_by_maturity, T)
local_vol, is_valid = dupire_local_volatility_from_total_variance(
    k_values, T, w, w_prime, w_double_prime, dw_dT=dw_dT, r=0.02
)
```

For detailed analysis and examples, see `explain_time_derivatives.py` and `test_realistic_local_vol.py`.

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
- Term structure models for time-dependent local volatility
- Stochastic local volatility (SLV) model implementations
