#!/usr/bin/env python3
"""
Local Volatility Examples

This script demonstrates how to compute and analyze local volatility from SVI and SSVI
implied volatility models using the Dupire formula.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from local_volatility import (
    compute_svi_local_volatility,
    compute_ssvi_local_volatility,
    compare_implied_vs_local_volatility,
    print_local_volatility_analysis,
    analyze_local_volatility_properties
)


def example_svi_local_volatility():
    """Example: Compute and analyze SVI local volatility."""
    print("=== SVI Local Volatility Example ===")
    
    # Parameters
    k_values = np.linspace(-1.5, 1.5, 200)
    T = 0.5
    a, b, rho, m, sigma = 0.04, 0.4, -0.3, 0.1, 0.3
    r = 0.05  # 5% interest rate
    
    # Compute local volatility
    local_vol, is_valid, diagnostics = compute_svi_local_volatility(
        k_values, T, a, b, rho, m, sigma, r=r
    )
    
    # Analyze results
    analysis = analyze_local_volatility_properties(
        k_values, local_vol, is_valid, "SVI"
    )
    print_local_volatility_analysis(analysis)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Implied vs Local Volatility
    implied_vol = np.sqrt(diagnostics['total_variance'] / T)
    
    ax1.plot(k_values, implied_vol, 'b-', linewidth=2, label='Implied Volatility')
    ax1.plot(k_values[is_valid], local_vol[is_valid], 'r-', linewidth=2, label='Local Volatility')
    if np.any(~is_valid):
        ax1.scatter(k_values[~is_valid], np.full(np.sum(~is_valid), np.nan), 
                   c='red', marker='x', s=50, label='Invalid regions')
    
    ax1.set_xlabel('Log-moneyness')
    ax1.set_ylabel('Volatility')
    ax1.set_title('SVI: Implied vs Local Volatility')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total Variance and its Derivatives
    ax2.plot(k_values, diagnostics['total_variance'], 'g-', linewidth=2, label='Total Variance w(k)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(k_values, diagnostics['first_derivative'], 'orange', linewidth=2, label="w'(k)")
    ax2_twin.plot(k_values, diagnostics['second_derivative'], 'purple', linewidth=2, label="w''(k)")
    
    ax2.set_xlabel('Log-moneyness')
    ax2.set_ylabel('Total Variance', color='g')
    ax2_twin.set_ylabel('Derivatives', color='orange')
    ax2.set_title('SVI Total Variance and Derivatives')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Local Vol vs Implied Vol Ratio
    valid_mask = is_valid & (implied_vol > 1e-6)
    if np.any(valid_mask):
        ratio = local_vol[valid_mask] / implied_vol[valid_mask]
        ax3.plot(k_values[valid_mask], ratio, 'purple', linewidth=2)
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Ratio = 1')
        ax3.set_xlabel('Log-moneyness')
        ax3.set_ylabel('Local Vol / Implied Vol')
        ax3.set_title('Local vs Implied Volatility Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print()


def example_ssvi_local_volatility():
    """Example: Compute and analyze SSVI local volatility."""
    print("=== SSVI Local Volatility Example ===")
    
    # Parameters
    k_values = np.linspace(-2.0, 2.0, 200)
    T = 0.25
    theta, phi, rho = 0.2, 1.0, -0.4
    r = 0.03
    
    # Compute local volatility
    local_vol, is_valid, diagnostics = compute_ssvi_local_volatility(
        k_values, T, theta, phi, rho, r=r
    )
    
    # Analyze results
    analysis = analyze_local_volatility_properties(
        k_values, local_vol, is_valid, "SSVI"
    )
    print_local_volatility_analysis(analysis)
    
    # Plot comparison
    implied_vol = np.sqrt(diagnostics['total_variance'] / T)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(k_values, implied_vol, 'b-', linewidth=2, label='SSVI Implied Vol')
    plt.plot(k_values[is_valid], local_vol[is_valid], 'r-', linewidth=2, label='SSVI Local Vol')
    if np.any(~is_valid):
        invalid_k = k_values[~is_valid]
        plt.axvspan(np.min(invalid_k), np.max(invalid_k), alpha=0.2, color='red', 
                   label='Invalid region')
    
    plt.xlabel('Log-moneyness')
    plt.ylabel('Volatility')
    plt.title('SSVI: Implied vs Local Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(k_values, diagnostics['first_derivative'], 'orange', linewidth=2, label="∂w/∂k")
    plt.plot(k_values, diagnostics['second_derivative'], 'purple', linewidth=2, label="∂²w/∂k²")
    plt.xlabel('Log-moneyness')
    plt.ylabel('Derivatives')
    plt.title('SSVI Total Variance Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print()


def example_model_comparison():
    """Example: Compare SVI and SSVI local volatilities."""
    print("=== SVI vs SSVI Local Volatility Comparison ===")
    
    k_values = np.linspace(-1.0, 1.0, 150)
    T = 0.5
    r = 0.02
    
    # Choose parameters to give similar ATM implied volatilities
    svi_params = {'a': 0.04, 'b': 0.4, 'rho': -0.25, 'm': 0.0, 'sigma': 0.3}
    ssvi_params = {'theta': 0.25, 'phi': 0.8, 'rho': -0.25}
    
    # Compare models
    comparison = compare_implied_vs_local_volatility(
        k_values, T, svi_params=svi_params, ssvi_params=ssvi_params, r=r
    )
    
    # Print analyses
    print_local_volatility_analysis(comparison['svi']['analysis'])
    print_local_volatility_analysis(comparison['ssvi']['analysis'])
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Implied volatilities
    ax = axes[0, 0]
    ax.plot(k_values, comparison['svi']['implied_vol'], 'b-', linewidth=2, label='SVI Implied')
    ax.plot(k_values, comparison['ssvi']['implied_vol'], 'g-', linewidth=2, label='SSVI Implied')
    ax.set_xlabel('Log-moneyness')
    ax.set_ylabel('Implied Volatility')
    ax.set_title('Implied Volatility Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Local volatilities
    ax = axes[0, 1]
    svi_valid = comparison['svi']['is_valid']
    ssvi_valid = comparison['ssvi']['is_valid']
    
    ax.plot(k_values[svi_valid], comparison['svi']['local_vol'][svi_valid], 
           'r-', linewidth=2, label='SVI Local')
    ax.plot(k_values[ssvi_valid], comparison['ssvi']['local_vol'][ssvi_valid], 
           'm-', linewidth=2, label='SSVI Local')
    
    ax.set_xlabel('Log-moneyness')
    ax.set_ylabel('Local Volatility')
    ax.set_title('Local Volatility Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ratios: Local/Implied
    ax = axes[1, 0]
    svi_ratio_mask = svi_valid & (comparison['svi']['implied_vol'] > 1e-6)
    ssvi_ratio_mask = ssvi_valid & (comparison['ssvi']['implied_vol'] > 1e-6)
    
    if np.any(svi_ratio_mask):
        svi_ratio = comparison['svi']['local_vol'][svi_ratio_mask] / comparison['svi']['implied_vol'][svi_ratio_mask]
        ax.plot(k_values[svi_ratio_mask], svi_ratio, 'r-', linewidth=2, label='SVI Ratio')
    
    if np.any(ssvi_ratio_mask):
        ssvi_ratio = comparison['ssvi']['local_vol'][ssvi_ratio_mask] / comparison['ssvi']['implied_vol'][ssvi_ratio_mask]
        ax.plot(k_values[ssvi_ratio_mask], ssvi_ratio, 'm-', linewidth=2, label='SSVI Ratio')
    
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Log-moneyness')
    ax.set_ylabel('Local Vol / Implied Vol')
    ax.set_title('Local/Implied Volatility Ratios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Difference: SVI Local - SSVI Local (where both valid)
    ax = axes[1, 1]
    both_valid = svi_valid & ssvi_valid
    if np.any(both_valid):
        diff = comparison['svi']['local_vol'][both_valid] - comparison['ssvi']['local_vol'][both_valid]
        ax.plot(k_values[both_valid], diff, 'purple', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Log-moneyness')
        ax.set_ylabel('SVI Local Vol - SSVI Local Vol')
        ax.set_title('Local Volatility Difference')
        ax.grid(True, alpha=0.3)
        
        print(f"Max local vol difference: {np.max(np.abs(diff)):.4f}")
        print(f"RMS local vol difference: {np.sqrt(np.mean(diff**2)):.4f}")
    
    plt.tight_layout()
    plt.show()
    print()


def example_sensitivity_analysis():
    """Example: Analyze sensitivity of local volatility to model parameters."""
    print("=== Local Volatility Sensitivity Analysis ===")
    
    k_values = np.linspace(-1.0, 1.0, 100)
    T = 0.5
    r = 0.03
    
    # Base SVI parameters
    base_params = {'a': 0.04, 'b': 0.4, 'rho': -0.3, 'm': 0.0, 'sigma': 0.3}
    
    # Parameter perturbations
    perturbations = {
        'rho': [-0.7, -0.5, -0.3, -0.1, 0.1],
        'b': [0.2, 0.3, 0.4, 0.5, 0.6],
        'sigma': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (param_name, values) in enumerate(perturbations.items()):
        ax = axes[i]
        
        for value in values:
            params = base_params.copy()
            params[param_name] = value
            
            local_vol, is_valid, _ = compute_svi_local_volatility(k_values, T, **params, r=r)
            
            if np.any(is_valid):
                ax.plot(k_values[is_valid], local_vol[is_valid], 
                       linewidth=2, label=f'{param_name}={value}')
        
        ax.set_xlabel('Log-moneyness')
        ax.set_ylabel('Local Volatility')
        ax.set_title(f'Sensitivity to {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print()


def main():
    """Run all local volatility examples."""
    print("Local Volatility Computation Examples")
    print("=" * 50)
    
    try:
        example_svi_local_volatility()
        example_ssvi_local_volatility()
        example_model_comparison()
        example_sensitivity_analysis()
        
        print("All local volatility examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
