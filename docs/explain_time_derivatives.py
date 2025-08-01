#!/usr/bin/env python3
"""
Comprehensive Analysis of Time Derivative Handling in Local Volatility

This script demonstrates the three different approaches for handling time derivatives
in the Dupire formula and shows their practical implications.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from local_volatility import dupire_local_volatility_from_total_variance
from svi_models import compute_svi_total_variance

def explain_time_derivatives():
    """
    Comprehensive explanation of time derivative handling in local volatility.
    """
    
    print("=" * 80)
    print("TIME DERIVATIVES IN LOCAL VOLATILITY: A DETAILED EXPLANATION")
    print("=" * 80)
    print()
    
    print("1. THE DUPIRE FORMULA STRUCTURE")
    print("-" * 40)
    print("σ_LV²(K,T) = (∂w/∂T + r·w) / (DENOMINATOR)")
    print()
    print("Where:")
    print("- w(k,T) = σ_IV²(K,T)·T is total variance")
    print("- ∂w/∂T is the time derivative of total variance")
    print("- r is the risk-free rate")
    print("- k = ln(K/S₀) is log-moneyness")
    print()
    
    print("2. THE TIME DERIVATIVE CHALLENGE")
    print("-" * 40)
    print("The time derivative ∂w/∂T is the most challenging part because:")
    print()
    print("THEORETICAL ASSUMPTION vs PRACTICAL REALITY:")
    print()
    print("❌ THEORETICAL (Time-Homogeneous):")
    print("   - Assume constant SVI parameters across all maturities")
    print("   - Then w(k,T) = T × w_svi(k) where w_svi(k) is SVI function")
    print("   - This gives ∂w/∂T = w_svi(k) = w(k,T)/T")
    print()
    print("✅ PRACTICAL (Slice-by-Slice Calibration):")
    print("   - Different SVI parameters for each maturity")
    print("   - Parameters come from fitting to market option prices")
    print("   - ∂w/∂T computed from neighboring slices using finite differences")
    print()
    
    print("3. IMPLEMENTATION APPROACHES")
    print("-" * 40)
    print()
    
    # Demonstrate different approaches
    k_values = np.linspace(-1.0, 1.0, 21)
    T = 1.0
    r = 0.02
    
    # Approach 1: Time-homogeneous (theoretical)
    print("APPROACH 1: Time-Homogeneous Model")
    print("Assumption: w(k,T) = T × w_svi(k)")
    print()
    svi_params = {'a': 0.04, 'b': 0.25, 'rho': -0.1, 'm': 0.0, 'sigma': 0.35}
    w_homogeneous = compute_svi_total_variance(k_values, **svi_params)
    dw_dT_homogeneous = w_homogeneous / T  # = w_svi(k)
    
    print(f"SVI parameters: {svi_params}")
    print(f"Total variance range: {np.min(w_homogeneous):.6f} - {np.max(w_homogeneous):.6f}")
    print(f"∂w/∂T = w/T range: {np.min(dw_dT_homogeneous):.6f} - {np.max(dw_dT_homogeneous):.6f}")
    print()
    
    # Approach 2: Slice-by-slice (realistic)
    print("APPROACH 2: Slice-by-Slice Calibration")
    print("Different parameters for different maturities")
    print()
    
    # Define realistic parameters for different maturities
    slices = {
        0.5: {'a': 0.02, 'b': 0.20, 'rho': -0.2, 'm': 0.02, 'sigma': 0.30},
        1.0: {'a': 0.04, 'b': 0.25, 'rho': -0.1, 'm': 0.00, 'sigma': 0.35},
        2.0: {'a': 0.08, 'b': 0.30, 'rho':  0.0, 'm': -0.02, 'sigma': 0.40}
    }
    
    for T_slice, params in slices.items():
        w_slice = compute_svi_total_variance(k_values, **params)
        print(f"T={T_slice:.1f}: w range = {np.min(w_slice):.6f} - {np.max(w_slice):.6f}")
    
    # Compute finite difference for ∂w/∂T at T=1.0
    w_before = compute_svi_total_variance(k_values, **slices[0.5])
    w_current = compute_svi_total_variance(k_values, **slices[1.0])
    w_after = compute_svi_total_variance(k_values, **slices[2.0])
    
    # Central difference
    dw_dT_realistic = (w_after - w_before) / (2.0 - 0.5)
    
    print(f"∂w/∂T from finite diff: {np.min(dw_dT_realistic):.6f} - {np.max(dw_dT_realistic):.6f}")
    print()
    
    print("4. COMPARISON OF TIME DERIVATIVES")
    print("-" * 40)
    
    ratio = dw_dT_realistic / dw_dT_homogeneous
    print(f"Ratio (realistic/homogeneous): {np.min(ratio):.4f} - {np.max(ratio):.4f}")
    print(f"Mean ratio: {np.mean(ratio):.4f}")
    print()
    
    if np.mean(ratio) < 0.9 or np.mean(ratio) > 1.1:
        print("⚠️  SIGNIFICANT DIFFERENCE detected!")
        print("   Slice-by-slice calibration produces different time derivatives")
        print("   than the time-homogeneous assumption.")
    else:
        print("✅ Time derivatives are similar between approaches")
    print()
    
    print("5. IMPACT ON LOCAL VOLATILITY")
    print("-" * 40)
    
    # Compute spatial derivatives (same for both since we're at T=1.0)
    from local_volatility import compute_svi_total_variance_derivatives
    w_current, dw_dk, d2w_dk2 = compute_svi_total_variance_derivatives(k_values, **slices[1.0])
    
    # Compute local volatility with both time derivatives
    local_vol_homogeneous, is_valid1 = dupire_local_volatility_from_total_variance(
        k_values, T, w_current, dw_dk, d2w_dk2, dw_dT=dw_dT_homogeneous, r=r
    )
    
    local_vol_realistic, is_valid2 = dupire_local_volatility_from_total_variance(
        k_values, T, w_current, dw_dk, d2w_dk2, dw_dT=dw_dT_realistic, r=r
    )
    
    # Compare results
    valid_both = is_valid1 & is_valid2
    if np.any(valid_both):
        lv1 = local_vol_homogeneous[valid_both]
        lv2 = local_vol_realistic[valid_both]
        
        print(f"Local vol (homogeneous): {np.min(lv1):.4f} - {np.max(lv1):.4f}")
        print(f"Local vol (realistic): {np.min(lv2):.4f} - {np.max(lv2):.4f}")
        
        diff_pct = 100 * (lv2 - lv1) / lv1
        print(f"Difference: {np.min(diff_pct):.2f}% - {np.max(diff_pct):.2f}%")
        print(f"Mean difference: {np.mean(diff_pct):.2f}%")
    print()
    
    print("6. CALENDAR ARBITRAGE IMPLICATIONS")
    print("-" * 40)
    print("When using constant SVI parameters across time:")
    print()
    
    # Check calendar arbitrage for constant parameters
    T_test = [0.25, 0.5, 1.0, 2.0]
    k_test = 0.0  # ATM
    constant_params = {'a': 0.04, 'b': 0.25, 'rho': -0.1, 'm': 0.0, 'sigma': 0.35}
    
    variances_constant = []
    for T_i in T_test:
        w_i = compute_svi_total_variance(np.array([k_test]), **constant_params)[0]
        variances_constant.append(w_i)
        print(f"T={T_i:.2f}: w={w_i:.6f}")
    
    # Check if increasing
    is_increasing = all(variances_constant[i] <= variances_constant[i+1] 
                       for i in range(len(variances_constant)-1))
    
    print(f"Calendar arbitrage free (constant params): {is_increasing}")
    
    if not is_increasing:
        print("❌ CALENDAR ARBITRAGE VIOLATION!")
        print("   Constant SVI parameters can violate no-arbitrage conditions")
    else:
        print("✅ No calendar arbitrage with constant parameters")
    print()
    
    # Compare with realistic (slice-by-slice) parameters
    print("With slice-by-slice calibration:")
    variances_realistic = []
    for T_i in T_test:
        if T_i in slices:
            w_i = compute_svi_total_variance(np.array([k_test]), **slices[T_i])[0]
            variances_realistic.append(w_i)
            print(f"T={T_i:.2f}: w={w_i:.6f}")
    
    is_increasing_realistic = all(variances_realistic[i] <= variances_realistic[i+1] 
                                 for i in range(len(variances_realistic)-1))
    print(f"Calendar arbitrage free (slice-by-slice): {is_increasing_realistic}")
    print()
    
    print("7. PRACTICAL RECOMMENDATIONS")
    print("-" * 40)
    print("✅ DO:")
    print("   - Use slice-by-slice calibration with different SVI parameters per maturity")
    print("   - Compute ∂w/∂T from neighboring slices using finite differences")
    print("   - Check calendar arbitrage conditions regularly")
    print("   - Use higher-order interpolation for smoother time derivatives")
    print()
    print("❌ AVOID:")
    print("   - Assuming ∂w/∂T = 0 (time-homogeneous)")
    print("   - Using constant SVI parameters across all maturities")
    print("   - Ignoring calendar arbitrage constraints")
    print()
    
    print("8. IMPLEMENTATION IN OUR CODE")
    print("-" * 40)
    print("Our implementation handles time derivatives as follows:")
    print()
    print("A. dupire_local_volatility_from_total_variance():")
    print("   - Accepts optional dw_dT parameter")
    print("   - If dw_dT=None: uses w/T (time-homogeneous assumption)")
    print("   - If dw_dT provided: uses the given time derivative")
    print()
    print("B. compute_time_derivative_from_slices():")
    print("   - Finds neighboring maturity slices")
    print("   - Computes finite difference (w2-w1)/(T2-T1)")
    print("   - Handles edge cases (boundary maturities)")
    print()
    print("C. test_realistic_local_vol.py:")
    print("   - Demonstrates practical slice-by-slice approach")
    print("   - Shows impact on local volatility computation")
    print("   - Includes calendar arbitrage checks")
    print()
    
    print("9. MATHEMATICAL INSIGHT")
    print("-" * 40)
    print("The time derivative ∂w/∂T represents how total variance changes with maturity.")
    print("This captures:")
    print("- Term structure effects in volatility")
    print("- Mean reversion in the underlying process")
    print("- Market expectations about future volatility")
    print()
    print("In constant volatility case: ∂w/∂T = σ² (constant)")
    print("In realistic markets: ∂w/∂T varies with strike and maturity")
    print()
    
    return {
        'homogeneous_dw_dT': dw_dT_homogeneous,
        'realistic_dw_dT': dw_dT_realistic,
        'local_vol_homogeneous': local_vol_homogeneous,
        'local_vol_realistic': local_vol_realistic,
        'calendar_arbitrage_constant': is_increasing,
        'calendar_arbitrage_realistic': is_increasing_realistic
    }

if __name__ == "__main__":
    results = explain_time_derivatives()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Time derivatives are CRUCIAL for correct local volatility computation.")
    print("The practical approach (slice-by-slice) often differs significantly")
    print("from the theoretical assumption (time-homogeneous).")
    print("Always verify calendar arbitrage conditions when using SVI/SSVI models.")
