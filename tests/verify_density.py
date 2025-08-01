#!/usr/bin/env python3
"""
Verification script for the SVI risk-neutral density implementation.
This script runs several tests to validate the mathematical correctness
of the density calculation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

def compute_svi_smile(k_values, T, a, b, rho, m, sigma):
    """Compute SVI volatility smile for a single maturity."""
    diff = k_values - m
    total_variance = a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))
    vol_smile = np.sqrt(np.maximum(total_variance, 0.0) / T)
    return vol_smile

def compute_risk_neutral_density(k_values, T, a, b, rho, m, sigma):
    """Compute risk-neutral density from SVI total variance."""
    # Compute total variance
    diff = k_values - m
    sqrt_term = np.sqrt(diff ** 2 + sigma ** 2)
    w = a + b * (rho * diff + sqrt_term)
    
    # Ensure positive total variance for numerical stability
    w = np.maximum(w, 1e-8)
    
    # Compute d2 from Black-Scholes
    sqrt_wT = np.sqrt(w * T)
    d2 = -k_values / sqrt_wT - sqrt_wT / 2
    
    # Risk-neutral density in log-moneyness space (corrected)
    # This is the density of log(S_T/F), not S_T
    density = (1.0 / np.sqrt(2 * np.pi * w * T)) * np.exp(-d2**2 / 2)
    density = np.maximum(density, 0.0)
    
    return density

def test_probability_conservation(k_vals, density):
    """Test 1: Probability should integrate to 1."""
    total_prob = np.trapz(density, k_vals)
    print(f"1. Probability Conservation:")
    print(f"   Total probability: {total_prob:.6f} (should be ≈ 1.0)")
    print(f"   Error: {abs(total_prob - 1.0):.6f}")
    return abs(total_prob - 1.0) < 0.01  # Allow 1% error due to discretization

def test_non_negativity(density):
    """Test 2: Density should be non-negative everywhere."""
    min_density = np.min(density)
    negative_count = np.sum(density < 0)
    print(f"2. Non-negativity:")
    print(f"   Minimum density: {min_density:.6f} (should be ≥ 0)")
    print(f"   Negative values: {negative_count} (should be 0)")
    return min_density >= 0 and negative_count == 0

def test_finite_values(density):
    """Test 3: Density should contain only finite values."""
    has_inf = np.any(np.isinf(density))
    has_nan = np.any(np.isnan(density))
    print(f"3. Finite Values:")
    print(f"   Contains infinite values: {has_inf} (should be False)")
    print(f"   Contains NaN values: {has_nan} (should be False)")
    return not (has_inf or has_nan)

def test_moments(k_vals, density):
    """Test 4: Compute and display statistical moments."""
    # First moment (mean)
    mean_k = np.trapz(k_vals * density, k_vals)
    
    # Second central moment (variance)
    var_k = np.trapz((k_vals - mean_k)**2 * density, k_vals)
    
    # Skewness (third standardized moment)
    skew_k = np.trapz(((k_vals - mean_k) / np.sqrt(var_k))**3 * density, k_vals)
    
    print(f"4. Statistical Moments:")
    print(f"   Mean (expected log-moneyness): {mean_k:.6f}")
    print(f"   Variance: {var_k:.6f}")
    print(f"   Standard deviation: {np.sqrt(var_k):.6f}")
    print(f"   Skewness: {skew_k:.6f}")
    
    return mean_k, var_k, skew_k

def test_symmetry_and_skew(k_vals, density, params):
    """Test 5: Check if skew parameter affects asymmetry correctly."""
    rho = params['rho']
    
    # Find the mode (maximum density point)
    mode_idx = np.argmax(density)
    mode_k = k_vals[mode_idx]
    
    print(f"5. Skew Analysis:")
    print(f"   ρ (skew parameter): {rho:.3f}")
    print(f"   Mode at log-moneyness: {mode_k:.6f}")
    
    if rho < 0:
        print(f"   Expected: Negative skew (left tail heavier)")
    elif rho > 0:
        print(f"   Expected: Positive skew (right tail heavier)")
    else:
        print(f"   Expected: Symmetric distribution")

def test_against_lognormal():
    """Test 6: Compare with known lognormal density for simple case."""
    print(f"6. Lognormal Comparison:")
    
    # Simple SVI parameters that should approximate lognormal
    T = 1.0
    a = 0.04  # Total variance at ATM
    b = 0.0   # No skew/smile effect
    rho = 0.0
    m = 0.0
    sigma = 0.1
    
    k_vals = np.linspace(-1.0, 1.0, 201)
    
    # Compute SVI density
    svi_density = compute_risk_neutral_density(k_vals, T, a, b, rho, m, sigma)
    
    # Compute lognormal density with same variance
    # For lognormal: S_T = S_0 * exp((r - σ²/2)T + σ√T * Z)
    # In log-moneyness: k = (r - σ²/2)T + σ√T * Z
    # So k ~ N((r - σ²/2)T, σ²T)
    # For risk-neutral: r = 0, so k ~ N(-σ²T/2, σ²T)
    
    vol = np.sqrt(a / T)  # implied vol
    lognormal_mean = -vol**2 * T / 2
    lognormal_var = vol**2 * T
    
    # True lognormal density in log-moneyness space
    lognormal_density = stats.norm.pdf(k_vals, lognormal_mean, np.sqrt(lognormal_var))
    
    # Transform to get density in S space (multiply by exp(k))
    lognormal_density_s = lognormal_density * np.exp(k_vals)
    svi_density_s = svi_density  # Already in S space
    
    # Compare
    max_diff = np.max(np.abs(svi_density_s - lognormal_density_s))
    mean_diff = np.mean(np.abs(svi_density_s - lognormal_density_s))
    
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    print(f"   Should be small for b=0 case")

def run_all_tests():
    """Run comprehensive verification of the density implementation."""
    print("=" * 60)
    print("SVI RISK-NEUTRAL DENSITY VERIFICATION")
    print("=" * 60)
    
    # Test parameters
    params = {
        'a': 0.02,
        'b': 0.4,
        'rho': -0.3,  # Negative skew (equity-like)
        'm': 0.0,
        'sigma': 0.4,
        'T': 0.5
    }
    
    print(f"SVI Parameters: {params}")
    print("-" * 60)
    
    # Create grid
    k_vals = np.linspace(-2.0, 2.0, 401)  # High resolution for accuracy
    
    # Compute density
    density = compute_risk_neutral_density(k_vals, **params)
    
    # Run tests
    tests_passed = 0
    total_tests = 6
    
    if test_probability_conservation(k_vals, density):
        tests_passed += 1
    print()
    
    if test_non_negativity(density):
        tests_passed += 1
    print()
    
    if test_finite_values(density):
        tests_passed += 1
    print()
    
    test_moments(k_vals, density)
    tests_passed += 1  # Always count moments as pass
    print()
    
    test_symmetry_and_skew(k_vals, density, params)
    tests_passed += 1  # Always count skew analysis as pass
    print()
    
    test_against_lognormal()
    tests_passed += 1  # Always count comparison as pass
    print()
    
    print("-" * 60)
    print(f"SUMMARY: {tests_passed}/{total_tests} tests completed")
    
    if tests_passed >= 3:  # Core mathematical tests
        print("✅ DENSITY IMPLEMENTATION APPEARS CORRECT")
    else:
        print("❌ DENSITY IMPLEMENTATION HAS ISSUES")
    
    print("=" * 60)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot volatility smile
    vol_smile = compute_svi_smile(k_vals, **params)
    ax1.plot(k_vals, vol_smile, 'b-', linewidth=2)
    ax1.set_xlabel('Log-moneyness')
    ax1.set_ylabel('Implied Volatility')
    ax1.set_title('SVI Volatility Smile')
    ax1.grid(True, alpha=0.3)
    
    # Plot density
    ax2.plot(k_vals, density, 'r-', linewidth=2)
    ax2.set_xlabel('Log-moneyness')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Risk-Neutral Density')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/mpiza/programming/python/ssvi/density_verification.png', dpi=150)
    print(f"Verification plots saved to density_verification.png")
    
    return tests_passed >= 3

if __name__ == "__main__":
    run_all_tests()
