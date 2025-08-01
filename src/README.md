# SSVI Source Code

This directory contains the core implementation modules for the SSVI project.

## Modules

### `parametric_ssvi.py`
The main parametric SSVI implementation with analytical derivatives. Provides:
- Time-dependent variance level θ(T) with mean reversion
- Rational function φ(θ) for skew parameterization
- Analytical derivatives for local volatility computation
- Comprehensive parameter validation

Key functions:
- `compute_parametric_ssvi_total_variance()`: Core total variance computation
- `compute_parametric_ssvi_local_volatility()`: Local volatility via Dupire formula
- `compute_parametric_ssvi_derivatives()`: All first and second derivatives
- `validate_parametric_ssvi_parameters()`: Parameter validation

### `svi_models.py`
Standard SVI and SSVI model implementations. Provides:
- Classic SVI parameterization
- SSVI (surface SVI) model
- Parameter validation utilities
- Surface generation functions

Key functions:
- `compute_svi_total_variance()`: Standard SVI total variance
- `compute_ssvi_total_variance()`: SSVI total variance
- `compute_svi_surface()`: Multi-maturity SVI surface
- `validate_svi_parameters()`: SVI parameter validation

### `local_volatility.py`
Local volatility computation utilities. Provides:
- Dupire formula implementations
- Numerical stability checks
- Local volatility analysis tools
- Comparison utilities

Key functions:
- `compute_svi_local_volatility()`: SVI local volatility
- `compute_ssvi_local_volatility()`: SSVI local volatility
- `dupire_local_volatility_from_total_variance()`: Generic Dupire implementation
- `analyze_local_volatility_properties()`: Analysis and validation

### `density_analysis.py`
Risk-neutral density analysis tools. Provides:
- Density computation from SVI models
- Arbitrage detection
- Moment calculations
- Density property verification

Key functions:
- `compute_svi_risk_neutral_density()`: Density from SVI parameters
- `verify_density_properties()`: Check positivity and integration
- `analyze_density_arbitrage()`: Arbitrage analysis
- `compute_moment_from_density()`: Moment calculations

## Usage

```python
# Import core modules
from src.parametric_ssvi import compute_parametric_ssvi_total_variance
from src.svi_models import compute_svi_total_variance
from src.local_volatility import compute_svi_local_volatility
from src.density_analysis import compute_svi_risk_neutral_density

# Or import everything
from src import *
```

## Mathematical Foundation

All implementations are based on the mathematical derivations documented in `../MATHEMATICAL_DERIVATIONS.md`. The analytical derivatives ensure:
- Machine precision accuracy (≈ 1e-12 relative error)
- Numerical stability
- Arbitrage-free surfaces
- Consistent local volatility computation

## Dependencies

- NumPy: Numerical computations
- SciPy: Optimization and special functions
- Matplotlib: Visualization (for analysis functions)
- Typing: Type hints
