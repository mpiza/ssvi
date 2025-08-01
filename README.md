# Volatility Surface Visualization Application

This application provides interactive visualization of volatility surfaces using the SVI (Stochastic Volatility Inspired) and SSVI (Surface SVI) parameterizations. It offers three different visualization modes to help understand how these models work and how their parameters affect volatility structures.

## Table of Contents

1. [Overview](#overview)
2. [Installation and Usage](#installation-and-usage)
3. [Visualization Modes](#visualization-modes)
4. [SVI Model](#svi-model)
5. [SSVI Model](#ssvi-model)
6. [Parameter Interpretations](#parameter-interpretations)
7. [Mathematical Background](#mathematical-background)
8. [Testing and Verification](#testing-and-verification)
9. [References](#references)

## Overview

Volatility surfaces are fundamental tools in quantitative finance for pricing and risk management of options. They describe how implied volatility varies across different strikes (moneyness) and maturities. This application implements two popular parameterizations:

- **SVI (Stochastic Volatility Inspired)**: A parametric model for volatility smiles at individual maturities
- **SSVI (Surface SVI)**: An extension that provides a coherent framework for the entire volatility surface

## Installation and Usage

### Prerequisites

```bash
pip install numpy matplotlib
```

### Running the Application

```bash
python volatility_surface_app.py
```

The application will prompt you to select one of three visualization modes:

1. **SVI Surface** - 3D surface plot using SVI parameterization
2. **SSVI Surface** - 3D surface plot using SSVI parameterization  
3. **SVI Smile** - 2D volatility smile at a single maturity

## Visualization Modes

### 1. SVI Surface (Option 1)

**Description**: Creates a 3D volatility surface by applying the same SVI parameters across all maturities and scaling implied volatility by √(w/T).

**Interactive Controls**:
- 5 sliders for SVI parameters: `a`, `b`, `ρ`, `m`, `σ`
- Reset button to restore default values
- 3D rotation and zoom capabilities

**Note**: This is a simplified approach where the same SVI smile shape is applied to all maturities. In reality, SVI parameters would vary with maturity.

### 2. SSVI Surface (Option 2)

**Description**: Displays a 3D volatility surface using the SSVI parameterization, which provides a more theoretically sound framework for modeling the entire surface.

**Interactive Controls**:
- 3 sliders for SSVI parameters: `θ`, `φ`, `ρ`
- Reset button to restore default values
- 3D rotation and zoom capabilities

### 3. SVI Smile (Option 3)

**Description**: Shows a 2D volatility smile for a single maturity using the SVI model, along with the corresponding risk-neutral density function. This provides the most accurate representation of how SVI was originally designed to work and reveals the market's implied probability distribution.

**Visualization**: Two-panel display:
- **Top panel**: SVI volatility smile vs. log-moneyness
- **Bottom panel**: Risk-neutral probability density derived from the smile

**Interactive Controls**:
- 5 sliders for SVI parameters: `a`, `b`, `ρ`, `m`, `σ`
- 1 slider for maturity: `T`
- Reset button to restore default values

**Advantages**: 
- Shows the true SVI behavior without cross-maturity artifacts
- Better for understanding individual parameter effects
- More pedagogically sound
- **NEW**: Displays the risk-neutral density, revealing market expectations about future price distributions

**Risk-Neutral Density**: The probability density function is computed using the Breeden-Litzenberger formula:
```
p(S) = e^(rT) × ∂²C/∂K²
```
In terms of log-moneyness k = ln(K/F), this becomes:
```
p(k) = e^k × (1/√(2πwT)) × exp(-d₂²/2)
```
where w(k) is the SVI total variance and d₂ = -k/√(wT) - √(wT)/2. This shows how the market prices different outcomes for the underlying asset.

## SVI Model

### Model Definition

The SVI (Stochastic Volatility Inspired) model, introduced by Gatheral (2004), parameterizes the total variance of a volatility smile as:

```
w(k) = a + b(ρ(k - m) + √((k - m)² + σ²))
```

Where:
- `k` = log-moneyness = ln(K/F), with K = strike, F = forward price
- `w(k)` = total variance at log-moneyness k
- Implied volatility: `σ_impl(k, T) = √(w(k)/T)`

### SVI Parameters

| Parameter | Range | Interpretation |
|-----------|-------|----------------|
| **a** | ≥ 0 | **Vertical shift**: Controls the overall level of volatility. Higher values shift the entire smile upward. |
| **b** | ≥ 0 | **Volatility of variance**: Controls the slope and overall "width" of the smile. Higher values make the smile steeper. |
| **ρ** | [-1, 1] | **Skew/correlation**: Controls the asymmetry of the smile. Negative values create typical equity-like skew (higher vol for puts). |
| **m** | ℝ | **Horizontal shift**: Shifts the smile left (m < 0) or right (m > 0) along the log-moneyness axis. |
| **σ** | > 0 | **Curvature**: Controls the curvature around the minimum. Higher values make the smile more "U-shaped". |

### Parameter Effects

- **a ↑**: Entire smile shifts up (higher volatilities across all strikes)
- **b ↑**: Smile becomes steeper and wider
- **ρ < 0**: Creates negative skew (typical for equity options)
- **ρ > 0**: Creates positive skew  
- **m ↑**: Smile shifts right (minimum moves to higher strikes)
- **σ ↑**: Smile becomes more curved around the minimum

## SSVI Model

### Model Definition

The SSVI (Surface SVI) model provides a parameterization for the entire volatility surface. The total variance is given by:

```
w(k, θ) = (θ/2)(1 + ρφk + √((φk + ρ)² + (1 - ρ²)))
```

Where:
- `θ > 0` represents the variance level
- `φ > 0` controls the slope
- `ρ ∈ (-1, 1)` controls the skew

### SSVI Parameters

| Parameter | Range | Interpretation |
|-----------|-------|----------------|
| **θ** | > 0 | **Variance level**: Controls the overall level of total variance. Roughly proportional to ATM variance. |
| **φ** | > 0 | **Slope parameter**: Controls how quickly volatility changes with log-moneyness. Higher values create steeper smiles. |
| **ρ** | (-1, 1) | **Skew parameter**: Controls the asymmetry. Negative values create equity-like skew patterns. |

### Advantages of SSVI

1. **Arbitrage-free**: When properly calibrated, SSVI surfaces are free of calendar spread arbitrage
2. **Parsimonious**: Only 3 parameters control the entire surface
3. **Tractable**: Closed-form expressions for many Greeks and exotic option prices
4. **Realistic**: Captures many stylized facts of equity volatility surfaces

## Mathematical Background

### Total Variance vs. Implied Volatility

The relationship between total variance `w` and implied volatility `σ_impl` is:

```
σ_impl(k, T) = √(w(k)/T)
```

This means:
- Total variance is what the models parameterize directly
- Implied volatility is derived by dividing by time and taking the square root
- For fixed total variance, implied volatility decreases with maturity (√T effect)

### Log-Moneyness

Log-moneyness is defined as:
```
k = ln(K/F)
```

Where:
- K = strike price
- F = forward price = S₀e^(rT) for stock options
- k = 0 corresponds to at-the-money (ATM)
- k < 0 corresponds to in-the-money calls (out-of-the-money puts)
- k > 0 corresponds to out-of-the-money calls (in-the-money puts)

### Risk-Neutral Density

The risk-neutral density function represents the market's implied probability distribution for the underlying asset at expiration. It's derived from option prices using the Breeden-Litzenberger formula:

```
p(S) = e^(rT) × ∂²C/∂K²
```

Where `C(K,T)` is the call option price. In terms of log-moneyness k = ln(K/F):

```
p(k) = e^k × (1/√(2πwT)) × exp(-d₂²/2)
```

Where w(k) is the total variance and d₂ = -k/√(wT) - √(wT)/2.

**Physical Interpretation**:
- **Shape**: Shows where the market expects the asset price to be at expiration
- **Mode**: The peak indicates the most likely outcome
- **Skewness**: Asymmetry reveals directional bias (equity markets typically show negative skew)
- **Tails**: Heavy tails indicate higher probability of extreme moves

**Parameter Effects on Density**:
- **a ↑**: Flattens the density (higher uncertainty)
- **b ↑**: Creates more pronounced skewness and tail behavior
- **ρ < 0**: Shifts probability mass toward lower prices (negative skew)
- **m ↑**: Shifts the entire distribution to higher prices
- **σ ↑**: Makes the distribution more peaked around the center

### Typical Volatility Smile Shapes

**Equity Options**:
- Negative skew (ρ < 0 in SVI)
- Higher volatility for low strikes (protective puts)
- "Volatility smile" or "smirk"

**FX Options**:
- More symmetric smiles
- Often positive correlation parameter

**Commodity Options**:
- Can show positive skew due to supply constraints

## Implementation Notes

### Surface Construction in This Application

**SVI Surface Mode**: The application creates a surface by:
1. Computing SVI total variance as a function of log-moneyness only
2. Replicating this across all maturities
3. Converting to implied volatility using σ = √(w/T)

**Limitations**: This approach assumes the same SVI parameters for all maturities, which is unrealistic. In practice, SVI parameters would be calibrated separately for each maturity.

**SSVI Surface Mode**: Uses the theoretical SSVI framework, which is more appropriate for surface modeling.

### Numerical Considerations

The application includes several numerical safeguards:
- Clipping of negative total variances to zero
- Avoiding division by zero in maturity
- Numerical stability in square root calculations

## Educational Use

This application is designed for educational purposes to help understand:

1. **How volatility models work**: Interactive parameter adjustment shows immediate effects
2. **Parameter intuition**: Each slider demonstrates what each parameter controls
3. **Model differences**: Comparison between SVI and SSVI approaches
4. **Smile vs. Surface**: Understanding the difference between single-maturity smiles and full surfaces
5. **Risk-neutral densities**: Visualizing how volatility smiles translate to probability distributions
6. **Market expectations**: Understanding what option prices reveal about market sentiment

**Key Learning Objectives**:
- **Volatility-Density Connection**: See how changes in volatility smile shape directly affect the implied probability distribution
- **Parameter Impact**: Understand how each SVI parameter affects both volatility and probability
- **Market Intuition**: Develop intuition for reading market expectations from option prices
- **Arbitrage Concepts**: Observe how certain parameter combinations can lead to unrealistic probability distributions

## Practical Applications

In practice, these models are used for:

- **Option pricing**: Interpolating volatilities for strikes/maturities not quoted in the market
- **Risk management**: Scenario analysis and Greeks calculation
- **Trading**: Identifying mispriced options relative to the fitted surface
- **Model calibration**: Fitting theoretical models to market prices

## Testing and Verification

The `tests/` directory contains comprehensive verification scripts for the mathematical implementation:

### Test Scripts

- **`compare_density_formulas.py`**: Compares old vs corrected SVI density implementations, showing dramatic improvement in probability conservation
- **`verify_density.py`**: Mathematical verification suite that validates density properties (integration to 1, non-negativity, etc.)
- **`test_extreme_svi.py`**: Tests robustness under extreme parameter values and boundary conditions

### Key Validation Results

The implementation has been rigorously tested and validates:
- ✅ Risk-neutral density integrates to ≈1.0 (probability conservation)
- ✅ No negative densities under normal parameter ranges (no false arbitrage)
- ✅ Proper mathematical shape function g(y) accounting for SVI surface curvature
- ✅ Numerical stability across parameter ranges

### Running Tests

The project includes comprehensive tests and examples:

```bash
# Core tests and verification
python tests/compare_density_formulas.py
python tests/verify_density.py  
python tests/test_extreme_svi.py
python tests/test_constant_volatility.py
python tests/test_realistic_local_vol.py

# Examples and documentation
python examples/basic_examples.py
python examples/local_volatility_examples.py
python docs/explain_time_derivatives.py
```

See `tests/README.md` for detailed documentation of each test script.

## Project Structure

```
ssvi/
├── README.md                    # Main documentation
├── volatility_surface_app.py    # Interactive visualization application
├── svi_models.py               # SVI and SSVI model implementations
├── local_volatility.py         # Local volatility computation using Dupire formula
├── density_analysis.py         # Risk-neutral density analysis
├── examples/                   # Usage examples and demonstrations
│   ├── basic_examples.py       # Basic SVI/SSVI examples
│   └── local_volatility_examples.py # Local volatility examples
├── tests/                      # Test suite and verification
│   ├── README.md              # Testing documentation
│   ├── test_constant_volatility.py # Constant volatility test case
│   ├── test_realistic_local_vol.py # Realistic local volatility test
│   ├── test_extreme_svi.py    # Edge case testing
│   └── verify_density.py      # Density verification tests
└── docs/                       # Documentation and educational materials
    └── explain_time_derivatives.py # Time derivative handling explanation
```

### Key Files

- **volatility_surface_app.py**: Main interactive application with 3D visualization
- **svi_models.py**: Core SVI/SSVI implementations and parameter validation
- **local_volatility.py**: Dupire formula implementation with multiple time derivative approaches
- **examples/**: Demonstrates practical usage patterns
- **tests/**: Comprehensive testing including edge cases and realistic scenarios
- **docs/**: Educational materials explaining mathematical concepts

## References

1. Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives." Presentation at Global Derivatives & Risk Management, Madrid.

2. Gatheral, J., & Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces." Quantitative Finance, 14(1), 59-71.

3. Martini, C., & Rossello, D. (2010). "Arbitrage-free implied volatility surfaces." arXiv preprint arXiv:1002.2834.

4. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide." John Wiley & Sons.

## License

This educational tool is provided as-is for learning purposes. Users should verify all calculations before using in any production environment.
