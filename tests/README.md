# SVI Volatility Surface Tests

This directory contains test scripts and verification tools for the SVI volatility surface implementation.

## Test Scripts

### `compare_density_formulas.py`
**Purpose**: Comprehensive comparison between the old (incorrect) and new (correct) SVI density implementations.

**What it does**:
- Compares old formula (missing shape function) vs correct formula (with shape function g(y))
- Shows dramatic improvement in probability conservation (from 1.447 to 0.996 integral)
- Demonstrates point-wise density corrections across log-moneyness range
- Validates that the correct formula eliminates false arbitrage opportunities

**Usage**: `python compare_density_formulas.py`

### `verify_density.py`
**Purpose**: Mathematical verification suite for the SVI risk-neutral density calculation.

**What it does**:
- Tests probability conservation (should integrate to ≈1.0)
- Checks for negative densities (arbitrage indicators)
- Validates numerical stability and finite values
- Computes expected log-moneyness and other statistical moments
- Comprehensive validation framework for density correctness

**Usage**: `python verify_density.py`

### `test_extreme_svi.py`
**Purpose**: Tests SVI density behavior under extreme parameter values.

**What it does**:
- Pushes SVI parameters to boundary conditions
- Tests numerical stability with extreme volatility levels
- Checks behavior with very high/low correlation values
- Validates density calculation robustness
- Identifies parameter ranges where arbitrage might appear

**Usage**: `python test_extreme_svi.py`

## Generated Files

### PNG Images
- `debug_density.png`: Density debugging visualizations
- `density_verification.png`: Density verification plots
- `svi_investigation.png`: SVI parameter investigation results

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
