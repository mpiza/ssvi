# SSVI Project Organization

This document describes the complete organization of the parametric SSVI project.

## Project Structure

```
ssvi/
├── src/                           # Core implementation package (Python package directory)
│   ├── __init__.py                # Package initialization with exports
│   ├── README.md                  # Module documentation
│   ├── parametric_ssvi.py         # Main parametric SSVI implementation
│   ├── svi_models.py              # Standard SVI and SSVI models
│   ├── local_volatility.py        # Local volatility computations
│   └── density_analysis.py        # Risk-neutral density analysis
│
├── apps/                          # Interactive applications
│   ├── __init__.py                # Package initialization
│   ├── README.md                  # Application documentation
│   ├── parametric_ssvi_iv_lv_density_app.py  # Main IV/LV/density app
│   ├── parametric_ssvi_app.py     # Comprehensive parametric SSVI app
│   ├── enhanced_parametric_ssvi_app.py       # Enhanced visualization app
│   ├── simple_parametric_ssvi_app.py         # Simplified demo app
│   └── volatility_surface_app.py  # Volatility surface visualization
│
├── tests/                         # Comprehensive test suite
│   ├── __init__.py                # Test package initialization
│   ├── README.md                  # Test documentation
│   ├── run_analytical_tests.py    # Test runner for analytical derivatives
│   ├── test_analytical_derivatives.py       # Analytical vs numerical tests
│   ├── test_phi_derivative_accuracy.py      # φ derivative accuracy tests
│   ├── test_parametric_ssvi.py    # Core parametric SSVI tests
│   ├── test_local_vol.py          # Local volatility tests
│   ├── test_constant_volatility.py           # Edge case tests
│   ├── test_realistic_local_vol.py           # Realistic parameter tests
│   ├── test_extreme_svi.py        # Extreme parameter tests
│   ├── compare_density_formulas.py           # Density formula comparisons
│   └── verify_density.py          # Density validation
│
├── analysis/                      # Analysis and performance tools
│   ├── __init__.py                # Package initialization
│   ├── README.md                  # Analysis documentation
│   ├── performance_analysis.py    # Performance comparison tools
│   ├── quick_iv_lv_comparison.py  # Quick IV/LV comparison
│   └── test_derivatives.py        # Derivative testing utilities
│
├── archive/                       # Legacy and reference materials
│   ├── __init__.py                # Package initialization
│   ├── README.md                  # Archive documentation
│   ├── IMPLEMENTATION_SUMMARY.py  # Development summary
│   ├── README_refactored.md       # Legacy documentation
│   └── REORGANIZATION_SUMMARY.md  # Reorganization history
│
├── docs/                          # Documentation directory
├── examples/                      # Example scripts and demos
├── .venv/                         # Virtual environment
├── __init__.py                    # Root package initialization
├── MATHEMATICAL_DERIVATIONS.md    # Complete mathematical documentation
├── PROJECT_ORGANIZATION.md        # Project structure documentation
├── ORGANIZATION_COMPLETE.md       # Organization completion notes
├── README.md                      # Main project documentation
├── setup.py                       # Setuptools-based project configuration
├── pyproject.toml                 # Modern Python project configuration (optional)
└── .gitignore                     # Git ignore rules
```

## Key Features

### Clean Root Directory
- Only directories and documentation files in root
- All Python modules organized into logical packages
- Professional project structure ready for distribution

### Core Implementation (`src/`)
- **parametric_ssvi.py**: Complete analytical implementation with derivatives
- **svi_models.py**: Standard SVI and SSVI utilities
- **local_volatility.py**: Dupire formula implementations
- **density_analysis.py**: Risk-neutral density tools

### Interactive Applications (`apps/`)
- **parametric_ssvi_iv_lv_density_app.py**: Main application with IV/LV/density plots
- Multiple specialized visualization applications
- All applications updated with proper import paths

### Comprehensive Testing (`tests/`)
- Full test coverage for all implementations
- Analytical vs numerical derivative comparisons
- Edge case and extreme parameter testing
- Density validation and arbitrage detection

### Analysis Tools (`analysis/`)
- Performance benchmarking utilities
- Derivative accuracy testing
- Quick comparison tools

## Import Usage

> **Note:** The recommended way to use the import examples below is to install the package in editable mode by running `pip install -e .` from the project root (ensure you have a `setup.py` or `pyproject.toml` in place). This avoids manual `PYTHONPATH` manipulation and ensures consistent imports across environments.

### Import Examples

#### During Development (when running from the project root, with `src/` in `PYTHONPATH`)
```python
from src.parametric_ssvi import compute_parametric_ssvi_total_variance
from src.svi_models import compute_svi_volatility_smile
from src.local_volatility import compute_svi_local_volatility
from src.density_analysis import compute_svi_risk_neutral_density
```

### Package-Level Imports
```python
# Import everything from src package
from src import *

# Specific module imports
import src.parametric_ssvi as pssvi
import src.svi_models as svi
```

## Development Benefits

1. **Professional Structure**: Clean, maintainable organization
2. **Easy Navigation**: Logical grouping of related functionality
3. **Import Safety**: Clear module paths prevent confusion
4. **Scalability**: Easy to add new modules or applications
5. **Documentation**: Comprehensive README files in each directory
6. **Testing**: Well-organized test suite with clear coverage

## Mathematical Foundation

All implementations are based on the complete mathematical derivations in `MATHEMATICAL_DERIVATIONS.md`, ensuring:
- Analytical derivatives with machine precision accuracy
- Arbitrage-free volatility surfaces
- Numerically stable local volatility computation
- Comprehensive parameter validation

This organization provides a solid foundation for further development, research, or production deployment of the parametric SSVI model.
