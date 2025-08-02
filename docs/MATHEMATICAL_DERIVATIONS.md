# Parametric SSVI Mathematical Derivations

This document provides comprehensive mathematical derivations for the parametric SSVI model implementation, including all derivatives and local volatility formulas.

## Table of Contents
1. [Parametric SSVI Model Framework](#1-parametric-ssvi-model-framework)
2. [Time-Dependent Variance Level θ(T)](#2-time-dependent-variance-level-θt)
3. [Rational Function φ(θ)](#3-rational-function-φθ)
4. [Total Variance Derivatives](#4-total-variance-derivatives)
5. [Time Derivative](#5-time-derivative)
6. [Local Volatility via Dupire Formula](#6-local-volatility-via-dupire-formula)
7. [Implementation Notes](#7-implementation-notes)

---

## 1. Parametric SSVI Model Framework

### 1.1 Basic SSVI Formula

The parametric SSVI model extends the standard SSVI parameterization with time-dependent parameters:

```
w(μ, T) = (θ(T)/2) * [1 + ρφ(θ(T))μ + √((φ(θ(T))μ + ρ)² + (1 - ρ²))]
```

Where:
- `w(μ, T)` is the total variance
- `μ` is the log-moneyness
- `T` is the time to maturity
- `ρ` is the correlation parameter (-1 < ρ < 1)
- `θ(T)` is the time-dependent variance level
- `φ(θ)` is a rational function controlling the skew

### 1.2 Parameter Structure

The model has the following parameters:
- **Correlation**: ρ ∈ (-1, 1)
- **Variance level**: θ∞ (long-term), θ₀ (initial), κ (mean reversion speed)
- **Rational function**: p₀, p₁, p₂ (numerator), q₀, q₁, q₂ (denominator)

---

## 2. Time-Dependent Variance Level θ(T)

### 2.1 Mean Reversion Formula

The variance level follows a mean-reverting process:

```
θ(T) = θ∞T + (θ₀ - θ∞) * (1 - e^(-κT))/κ
```

### 2.2 Limiting Cases

**Case 1: κ → 0 (No mean reversion)**
```
lim(κ→0) θ(T) = θ₀T
```

**Case 2: κ → ∞ (Instant mean reversion)**
```
lim(κ→∞) θ(T) = θ∞T
```

### 2.3 Time Derivative of θ(T)

```
∂θ(T)/∂T = θ∞ + (θ₀ - θ∞) * e^(-κT)
```

**Derivation:**
```
θ(T) = θ∞T + (θ₀ - θ∞) * (1 - e^(-κT))/κ

∂θ/∂T = θ∞ + (θ₀ - θ∞) * ∂/∂T[(1 - e^(-κT))/κ]
       = θ∞ + (θ₀ - θ∞) * (1/κ) * κe^(-κT)
       = θ∞ + (θ₀ - θ∞) * e^(-κT)
```

---

## 3. Rational Function φ(θ)

### 3.1 Definition

```
φ(θ) = (p₀ + p₁θ + p₂θ²)/(q₀ + q₁θ + q₂θ²)
```

where q₀ = 1 is fixed for identifiability.

### 3.2 Derivative with Respect to θ

```
∂φ/∂θ = [P'(θ)Q(θ) - P(θ)Q'(θ)] / [Q(θ)]²
```

Where:
- `P(θ) = p₀ + p₁θ + p₂θ²`
- `Q(θ) = 1 + q₁θ + q₂θ²`
- `P'(θ) = p₁ + 2p₂θ`
- `Q'(θ) = q₁ + 2q₂θ`

**Expanded form:**
```
∂φ/∂θ = [(p₁ + 2p₂θ)(1 + q₁θ + q₂θ²) - (p₀ + p₁θ + p₂θ²)(q₁ + 2q₂θ)] / (1 + q₁θ + q₂θ²)²
```

---

## 4. Total Variance Derivatives

### 4.1 First Derivative ∂w/∂μ

**Starting from:**
```
w(μ, T) = (θ_T/2) * [1 + ρφ_T μ + √((φ_T μ + ρ)² + (1 - ρ²))]
```

Let `z = φ_T μ + ρ`, then:
```
w(μ, T) = (θ_T/2) * [1 + ρφ_T μ + √(z² + (1 - ρ²))]
```

**Derivative:**
```
∂w/∂μ = (θ_T/2) * [ρφ_T + φ_T * z/√(z² + (1 - ρ²))]
       = (θ_T/2) * φ_T * [ρ + z/√(z² + (1 - ρ²))]
```

**Derivation steps:**
1. `∂/∂μ[1] = 0`
2. `∂/∂μ[ρφ_T μ] = ρφ_T`
3. `∂/∂μ[√(z² + (1 - ρ²))] = (1/2) * (z² + (1 - ρ²))^(-1/2) * 2z * φ_T = φ_T * z/√(z² + (1 - ρ²))`

### 4.2 Second Derivative ∂²w/∂μ²

**From the first derivative:**
```
∂w/∂μ = (θ_T/2) * φ_T * [ρ + z/√(z² + (1 - ρ²))]
```

**Second derivative:**
```
∂²w/∂μ² = (θ_T/2) * φ_T² * ∂/∂μ[z/√(z² + (1 - ρ²))]
```

**Computing the derivative of z/√(z² + (1 - ρ²)):**

Let `f(z) = z/√(z² + c)` where `c = 1 - ρ²`

```
f'(z) = [√(z² + c) - z * z/√(z² + c)] / (z² + c)
      = [√(z² + c) - z²/√(z² + c)] / (z² + c)
      = [(z² + c) - z²] / [(z² + c)^(3/2)]
      = c / (z² + c)^(3/2)
      = (1 - ρ²) / (z² + (1 - ρ²))^(3/2)
```

Since `∂z/∂μ = φ_T`:

```
∂²w/∂μ² = (θ_T/2) * φ_T² * (1 - ρ²) / (z² + (1 - ρ²))^(3/2)
```

---

## 5. Time Derivative

### 5.1 Chain Rule Application

The total variance depends on time through both θ(T) and φ(θ(T)):

```
∂w/∂T = ∂w/∂θ_T * ∂θ_T/∂T + ∂w/∂φ_T * ∂φ_T/∂T
```

Where `∂φ_T/∂T = ∂φ/∂θ * ∂θ_T/∂T`

### 5.2 Partial Derivatives

**∂w/∂θ_T:**
```
w = (θ_T/2) * [1 + ρφ_T μ + √((φ_T μ + ρ)² + (1 - ρ²))]

∂w/∂θ_T = (1/2) * [1 + ρφ_T μ + √((φ_T μ + ρ)² + (1 - ρ²))]
```

**∂w/∂φ_T:**
```
∂w/∂φ_T = (θ_T/2) * [ρμ + μ * (φ_T μ + ρ)/√((φ_T μ + ρ)² + (1 - ρ²))]
         = (θ_T/2) * μ * [ρ + (φ_T μ + ρ)/√((φ_T μ + ρ)² + (1 - ρ²))]
         = (θ_T/2) * μ * [ρ + z/√(z² + (1 - ρ²))]
```

### 5.3 Complete Time Derivative

```
∂w/∂T = (1/2) * [1 + ρφ_T μ + √(z² + (1 - ρ²))] * ∂θ_T/∂T
       + (θ_T/2) * μ * [ρ + z/√(z² + (1 - ρ²))] * ∂φ/∂θ * ∂θ_T/∂T
```

---

## 6. Local Volatility via Dupire Formula

### 6.1 Dupire Formula for Total Variance

The local volatility squared is given by:

```
σ_LV²(μ, T) = (∂w/∂T + r * w) / [1 - μ/(2w) * ∂w/∂μ - (1/4) * ((1/w) * (∂w/∂μ)² + ∂²w/∂μ²)]
```

### 6.2 Numerator

```
Numerator = ∂w/∂T + r * w
```

Where r is the risk-free rate.

### 6.3 Denominator

```
Denominator = 1 - μ/(2w) * ∂w/∂μ - (1/4) * ((1/w) * (∂w/∂μ)² + ∂²w/∂μ²)
```

**Breaking down the terms:**
1. **Linear term**: `1`
2. **First-order correction**: `-μ/(2w) * ∂w/∂μ`
3. **Second-order correction**: `(1/4) * ((1/w) * (∂w/∂μ)² + ∂²w/∂μ²)`

### 6.4 Stability Considerations

The local volatility computation can be numerically unstable when:
- The denominator approaches zero
- The total variance w is very small
- The derivatives have extreme values

**Numerical safeguards:**
1. Ensure w > ε for small ε (e.g., 1e-12)
2. Check denominator > ε
3. Validate that σ_LV² > 0

---

## 7. Implementation Notes

### 7.1 Analytical Derivative for ∂φ/∂θ

The derivative ∂φ/∂θ is computed analytically using the quotient rule:

```
∂φ/∂θ = [P'(θ)Q(θ) - P(θ)Q'(θ)] / [Q(θ)]²
```

Where:
- `P(θ) = p₀ + p₁θ + p₂θ²`
- `Q(θ) = 1 + q₁θ + q₂θ²`
- `P'(θ) = p₁ + 2p₂θ`
- `Q'(θ) = q₁ + 2q₂θ`

This provides exact derivatives with machine precision accuracy, avoiding numerical differentiation errors.

### 7.2 Parameter Validation

**Critical constraints:**
1. `-1 < ρ < 1` (correlation bounds)
2. `θ∞, θ₀ > 0` (positive variance levels)
3. `κ ≥ 0` (non-negative mean reversion)
4. `φ(θ) > 0` for all relevant θ values
5. Denominator of φ(θ) ≠ 0

### 7.3 Edge Cases

**κ → 0 case:**
When κ is sufficiently small (e.g., κ < 1e-6), use the approximation `θ(T) = θ₀ * T` and `∂θ/∂T = θ₀`.

**Small denominator in φ(θ):**
Replace with small positive value (e.g., 1e-12) and issue warning

### 7.4 Computational Complexity

The analytical approach provides:
- **Accuracy**: Machine precision derivatives (≈ 1e-12 relative error)
- **Speed**: O(n) complexity for n grid points
- **Stability**: No finite difference errors
- **Consistency**: Exact derivatives ensure arbitrage-free surfaces

---

## References

1. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*
2. Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces
3. Dupire, B. (1994). Pricing with a smile
4. Hendriks, A. (2021). Parametric models for volatility surfaces

---

*This document provides the complete mathematical foundation for the parametric SSVI implementation with analytical derivatives.*
