#!/usr/bin/env python3
"""
Realistic test of local volatility computation using slice-by-slice calibration.

This test demonstrates how local volatility should be computed in practice:
1. Different SVI parameters for different maturities (slice-by-slice calibration)
2. Time derivatives computed from neighboring time slices
3. Proper handling of calendar arbitrage constraints
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.svi_models import compute_svi_volatility_smile, compute_svi_total_variance
from src.local_volatility import compute_svi_total_variance_derivatives, dupire_local_volatility_from_total_variance

def create_realistic_svi_surface():
    """
    Create a realistic SVI surface with different parameters for each maturity.
    
    In practice, each maturity slice is calibrated independently to market data,
    resulting in different SVI parameters for each slice.
    """
    # Define maturity grid
    maturities = np.array([0.25, 0.5, 1.0, 2.0])  # 3M, 6M, 1Y, 2Y
    
    # Different SVI parameters for each maturity (realistic calibration)
    # These would typically come from fitting to market option prices
    svi_params_by_maturity = {
        0.25: {'a': 0.01, 'b': 0.15, 'rho': -0.3, 'm': 0.05, 'sigma': 0.25},  # 3M
        0.5:  {'a': 0.02, 'b': 0.20, 'rho': -0.2, 'm': 0.02, 'sigma': 0.30},  # 6M  
        1.0:  {'a': 0.04, 'b': 0.25, 'rho': -0.1, 'm': 0.00, 'sigma': 0.35},  # 1Y
        2.0:  {'a': 0.08, 'b': 0.30, 'rho':  0.0, 'm': -0.02, 'sigma': 0.40}, # 2Y
    }
    
    return maturities, svi_params_by_maturity

def compute_time_derivative_from_slices(k_values, maturities, svi_params_by_maturity, target_T):
    """
    Compute ∂w/∂T by finite differences from neighboring maturity slices.
    
    This is how it's done in practice: look at total variance at neighboring
    maturities and compute the finite difference.
    """
    # Find the closest maturities to target_T
    idx = np.searchsorted(maturities, target_T)
    
    if idx == 0:
        # Use forward difference
        T1, T2 = maturities[0], maturities[1]
        params1 = svi_params_by_maturity[T1]
        params2 = svi_params_by_maturity[T2]
    elif idx >= len(maturities):
        # Use backward difference
        T1, T2 = maturities[-2], maturities[-1]
        params1 = svi_params_by_maturity[T1]
        params2 = svi_params_by_maturity[T2]
    else:
        # Use central difference if possible, otherwise forward/backward
        T1, T2 = maturities[idx-1], maturities[idx]
        params1 = svi_params_by_maturity[T1]
        params2 = svi_params_by_maturity[T2]
    
    # Compute total variance at both maturities
    w1 = compute_svi_total_variance(k_values, **params1)
    w2 = compute_svi_total_variance(k_values, **params2)
    
    # Finite difference approximation to ∂w/∂T
    dw_dT = (w2 - w1) / (T2 - T1)
    
    # Interpolate/extrapolate to target maturity if needed
    if target_T != T1 and target_T != T2:
        # Linear interpolation/extrapolation
        alpha = (target_T - T1) / (T2 - T1)
        w_target = (1 - alpha) * w1 + alpha * w2
        # Note: For simplicity, we use the same dw_dT
        # In practice, you might use higher-order interpolation
    else:
        w_target = w1 if target_T == T1 else w2
    
    return w_target, dw_dT

def test_realistic_local_volatility():
    """Test local volatility with realistic slice-by-slice calibration."""
    
    print("=== Realistic Local Volatility Test ===")
    print("Using slice-by-slice SVI calibration with time derivatives from neighboring slices")
    print()
    
    # Create realistic SVI surface
    maturities, svi_params_by_maturity = create_realistic_svi_surface()
    
    print("SVI parameters by maturity:")
    for T, params in svi_params_by_maturity.items():
        print(f"T={T:.2f}: a={params['a']:.3f}, b={params['b']:.3f}, rho={params['rho']:.2f}, "
              f"m={params['m']:.3f}, sigma={params['sigma']:.3f}")
    print()
    
    # Test at intermediate maturity
    target_T = 1.0
    r = 0.02
    k_values = np.linspace(-2.0, 2.0, 101)
    
    print(f"Computing local volatility at T = {target_T:.2f} years")
    
    # Get parameters for this maturity
    target_params = svi_params_by_maturity[target_T]
    
    # Compute implied volatility at target maturity
    implied_vol = compute_svi_volatility_smile(k_values, target_T, **target_params)
    
    # Compute spatial derivatives at target maturity
    w_target, dw_dk, d2w_dk2 = compute_svi_total_variance_derivatives(k_values, **target_params)
    
    # Compute time derivative from neighboring slices
    w_from_slices, dw_dT = compute_time_derivative_from_slices(
        k_values, maturities, svi_params_by_maturity, target_T
    )
    
    # Verify consistency
    w_direct = compute_svi_total_variance(k_values, **target_params)
    print(f"Total variance consistency check:")
    print(f"  Max difference: {np.max(np.abs(w_target - w_direct)):.8f}")
    print(f"  From slices vs direct: {np.max(np.abs(w_from_slices - w_direct)):.8f}")
    print()
    
    # Compute local volatility using the realistic time derivative
    local_vol, is_valid = dupire_local_volatility_from_total_variance(
        k_values, target_T, w_target, dw_dk, d2w_dk2, dw_dT=dw_dT, r=r
    )
    
    valid_count = np.sum(is_valid)
    print(f"Local volatility computation:")
    print(f"  Valid points: {valid_count}/{len(k_values)} ({100*valid_count/len(k_values):.1f}%)")
    
    if valid_count > 0:
        valid_local = local_vol[is_valid]
        valid_implied = implied_vol[is_valid]
        
        print(f"  Implied vol range: {np.min(valid_implied):.4f} - {np.max(valid_implied):.4f}")
        print(f"  Local vol range: {np.min(valid_local):.4f} - {np.max(valid_local):.4f}")
        
        # Analyze the relationship
        ratio = valid_local / valid_implied
        print(f"  Local/Implied ratio: {np.min(ratio):.4f} - {np.max(ratio):.4f}")
        print(f"  Mean ratio: {np.mean(ratio):.4f}")
    
    print()
    
    # Check time derivative behavior
    print("Time derivative analysis:")
    print(f"  ∂w/∂T range: {np.min(dw_dT):.6f} - {np.max(dw_dT):.6f}")
    print(f"  w/T range: {np.min(w_target/target_T):.6f} - {np.max(w_target/target_T):.6f}")
    print(f"  ∂w/∂T vs w/T ratio: {np.min(dw_dT/(w_target/target_T)):.4f} - {np.max(dw_dT/(w_target/target_T)):.4f}")
    print()
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Implied vs Local volatility
    plt.subplot(2, 3, 1)
    plt.plot(k_values, implied_vol, 'b-', linewidth=2, label='Implied volatility')
    if valid_count > 0:
        plt.plot(k_values[is_valid], local_vol[is_valid], 'g-', linewidth=2, label='Local volatility')
    plt.xlabel('Log-moneyness')
    plt.ylabel('Volatility')
    plt.title(f'Implied vs Local Volatility (T={target_T:.1f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Total variance and time derivative
    plt.subplot(2, 3, 2)
    plt.plot(k_values, w_target, 'b-', linewidth=2, label='Total variance w(k,T)')
    plt.plot(k_values, dw_dT, 'r-', linewidth=2, label='∂w/∂T from slices')
    plt.plot(k_values, w_target/target_T, 'g--', linewidth=2, label='w/T (constant vol case)')
    plt.xlabel('Log-moneyness')
    plt.ylabel('Total variance')
    plt.title('Total Variance and Time Derivative')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Spatial derivatives
    plt.subplot(2, 3, 3)
    plt.plot(k_values, dw_dk, 'g-', linewidth=2, label='∂w/∂k')
    plt.plot(k_values, d2w_dk2, 'r-', linewidth=2, label='∂²w/∂k²')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Log-moneyness')
    plt.ylabel('Derivatives')
    plt.title('Spatial Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Volatility surface across maturities
    plt.subplot(2, 3, 4)
    k_plot = np.linspace(-1.5, 1.5, 50)
    for T in maturities:
        params = svi_params_by_maturity[T]
        vol_smile = compute_svi_volatility_smile(k_plot, T, **params)
        plt.plot(k_plot, vol_smile, linewidth=2, label=f'T={T:.2f}')
    plt.xlabel('Log-moneyness')
    plt.ylabel('Implied volatility')
    plt.title('SVI Volatility Surface (All Maturities)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Local vs Implied ratio
    plt.subplot(2, 3, 5)
    if valid_count > 0:
        ratio = local_vol[is_valid] / implied_vol[is_valid]
        plt.plot(k_values[is_valid], ratio, 'purple', linewidth=2)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Ratio = 1')
        plt.xlabel('Log-moneyness')
        plt.ylabel('Local / Implied')
        plt.title('Local vs Implied Volatility Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Calendar arbitrage check
    plt.subplot(2, 3, 6)
    k_test = np.linspace(-1.0, 1.0, 21)
    total_variances = []
    for T in maturities:
        params = svi_params_by_maturity[T]
        w = compute_svi_total_variance(k_test, **params)
        total_variances.append(w)
        plt.plot(k_test, w, 'o-', linewidth=2, label=f'T={T:.2f}')
    
    # Check for calendar arbitrage (total variance should be increasing in time)
    plt.xlabel('Log-moneyness')
    plt.ylabel('Total variance')
    plt.title('Calendar Arbitrage Check')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calendar arbitrage analysis
    print("Calendar arbitrage analysis:")
    for i, k in enumerate([0.0]):  # Check at-the-money
        k_idx = np.argmin(np.abs(k_test - k))
        variances_at_k = [tv[k_idx] for tv in total_variances]
        is_increasing = all(variances_at_k[i] <= variances_at_k[i+1] for i in range(len(variances_at_k)-1))
        print(f"  At k={k:.1f}: Total variance increasing with time: {is_increasing}")
        if not is_increasing:
            print(f"    Values: {[f'{v:.6f}' for v in variances_at_k]}")

if __name__ == "__main__":
    test_realistic_local_volatility()
