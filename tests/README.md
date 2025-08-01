# SVI Volatility Surface Tests

This directory contains test scripts and verification tools for the SVI volatility surface implementation.

## Test Scripts

### Core SVI Tests

#### `compare_density_formulas.py`
**Purpose**: Comprehensive comparison between the old (incorrect) and new (correct) SVI density implementations.

**What it does**:
- Compares old formula (missing shape function) vs correct formula (with shape function g(y))
- Shows dramatic improvement in probability conservation (from 1.447 to 0.996 integral)
- Demonstrates point-wise density corrections across log-moneyness range
- Validates that the correct formula eliminates false arbitrage opportunities

**Usage**: `python compare_density_formulas.py`

#### `verify_density.py`
**Purpose**: Mathematical verification suite for the SVI risk-neutral density calculation.

**What it does**:
- Tests probability conservation (should integrate to ≈1.0)
- Checks for negative densities (arbitrage indicators)
- Validates numerical stability and finite values
- Computes expected log-moneyness and other statistical moments
- Comprehensive validation framework for density correctness

**Usage**: `python verify_density.py`

#### `test_extreme_svi.py`
**Purpose**: Tests SVI density behavior under extreme parameter values.

**What it does**:
- Pushes SVI parameters to boundary conditions
- Tests numerical stability with extreme volatility levels
- Checks behavior with very high/low correlation values
- Validates density calculation robustness
- Identifies parameter ranges where arbitrage might appear

**Usage**: `python test_extreme_svi.py`

### Parametric SSVI Tests

#### `test_parametric_ssvi.py`
**Purpose**: Basic parametric SSVI model functionality tests.

#### `test_analytical_derivatives.py`
**Purpose**: Comprehensive test of parametric SSVI with analytical derivatives.

**What it does**:
- Validates all derivative calculations (∂w/∂μ, ∂²w/∂μ², ∂w/∂T)
- Checks positivity constraints for arbitrage-free surfaces
- Generates visualizations of total variance and derivatives
- Tests complete parametric SSVI framework

**Usage**: `python test_analytical_derivatives.py`

#### `test_phi_derivative_accuracy.py`
**Purpose**: Verification that analytical φ(θ) derivative matches numerical derivatives.

**What it does**:
- Compares analytical quotient rule implementation vs numerical differentiation
- Tests accuracy across different θ values and step sizes
- Validates machine precision accuracy of analytical approach
- Generates accuracy plots and error analysis

**Usage**: `python test_phi_derivative_accuracy.py`

### Local Volatility Tests

#### `test_constant_volatility.py`
**Purpose**: Validates Dupire formula implementation using constant volatility case.

#### `test_realistic_local_vol.py`
**Purpose**: Tests local volatility computation with realistic market-like parameters.

#### `test_local_vol.py`
**Purpose**: General local volatility testing framework.

## Directory Structure

```
tests/
├── README.md                           # This file
├── plots/                              # Generated plots directory
│   ├── parametric_ssvi_derivatives_test.png
│   ├── phi_derivative_verification.png
│   ├── debug_density.png
│   ├── density_verification.png
│   └── svi_investigation.png
├── compare_density_formulas.py         # SVI density formula comparison
├── verify_density.py                   # SVI density verification suite
├── test_extreme_svi.py                 # Extreme parameter testing
├── test_parametric_ssvi.py             # Basic parametric SSVI tests
├── test_analytical_derivatives.py      # Analytical derivatives validation
├── test_phi_derivative_accuracy.py     # φ(θ) derivative accuracy test
├── test_constant_volatility.py         # Constant volatility Dupire test
├── test_realistic_local_vol.py         # Realistic local vol testing
└── test_local_vol.py                   # General local volatility tests
```

## Running Tests

### Individual Tests
```bash
cd tests
python test_analytical_derivatives.py      # Test parametric SSVI derivatives
python test_phi_derivative_accuracy.py     # Test φ derivative accuracy
python verify_density.py                   # Verify SVI density calculations
```

### All Tests
```bash
cd tests
for test_file in test_*.py; do
    echo "Running $test_file..."
    python "$test_file"
done
```

## Key Mathematical Results

The tests validate the complete SVI density formula:

```
w(y) = a + b[ρ(y-m) + √((y-m)² + σ²)]
w'(y) = b[ρ + (y-m)/√((y-m)² + σ²)]
w''(y) = b σ² / ((y-m)² + σ²)^(3/2)

g(y) = [1 - y w'(y)/(2w(y))]² - (w'(y))²/4 [1/w(y) + 1/4] + w''(y)/2

p(y) = g(y)/√(2πw(y)) * exp(-d₋²/2)
where d₋ = -y/√w(y) - √w(y)/2
```

## Test Results Summary

- ✅ Density integrates to ≈1.0 (probability conservation)
- ✅ No negative densities under normal parameter ranges
- ✅ Proper mathematical properties (finite, non-NaN values)
- ✅ Dramatic improvement over simplified Black-Scholes density
- ✅ Shape function g(y) correctly accounts for SVI surface curvature

## Running All Tests

To run all verification tests:
```bash
cd tests/
python compare_density_formulas.py
python verify_density.py
python test_extreme_svi.py
```
