#!/usr/bin/env python3
"""
Example usage of the refactored SVI models and density analysis modules.

This script demonstrates how to use the separated core calculation functions
independently of the interactive visualization application.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the core calculation modules
from svi_models import (
    compute_svi_volatility_smile,
    compute_svi_surface,
    compute_ssvi_surface,
    validate_svi_parameters,
    validate_ssvi_parameters
)
from density_analysis import (
    compute_svi_risk_neutral_density,
    verify_density_properties,
    analyze_density_arbitrage
)


def example_svi_smile():
    """Example: Compute and plot an SVI volatility smile."""
    print("=== SVI Volatility Smile Example ===")
    
    # Define parameters
    k_values = np.linspace(-2.0, 2.0, 100)
    T = 0.5  # 6 months
    a, b, rho, m, sigma = 0.02, 0.4, -0.2, 0.0, 0.4
    
    # Validate parameters
    is_valid, violations = validate_svi_parameters(a, b, rho, m, sigma)
    print(f"Parameters valid: {is_valid}")
    if violations:
        for violation in violations:
            print(f"  - {violation}")
    
    # Compute smile
    vol_smile = compute_svi_volatility_smile(k_values, T, a, b, rho, m, sigma)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, vol_smile, 'b-', linewidth=2, label='SVI Smile')
    plt.xlabel('Log-moneyness')
    plt.ylabel('Implied Volatility')
    plt.title(f'SVI Volatility Smile (T={T} years)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print(f"ATM volatility: {vol_smile[len(vol_smile)//2]:.4f}")
    print()


def example_density_analysis():
    """Example: Analyze risk-neutral density from SVI model."""
    print("=== Risk-Neutral Density Analysis Example ===")
    
    # Define parameters - use some potentially problematic ones
    k_values = np.linspace(-3.0, 3.0, 300)
    T = 0.25  # 3 months
    a, b, rho, m, sigma = 0.05, 0.8, -0.7, 0.2, 0.3
    
    # Compute density
    density = compute_svi_risk_neutral_density(k_values, T, a, b, rho, m, sigma)
    
    # Analyze for arbitrage
    analysis = analyze_density_arbitrage(k_values, density, T, a, b, rho, m, sigma)
    
    print(f"Arbitrage detected: {analysis['has_arbitrage']}")
    print(f"Arbitrage score: {analysis['arbitrage_score']:.4f}")
    
    # Detailed verification
    verification = verify_density_properties(k_values, density, verbose=True)
    
    # Plot density
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    vol_smile = compute_svi_volatility_smile(k_values, T, a, b, rho, m, sigma)
    plt.plot(k_values, vol_smile, 'b-', linewidth=2)
    plt.xlabel('Log-moneyness')
    plt.ylabel('Implied Volatility')
    plt.title('SVI Volatility Smile')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(k_values, density, 'r-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Log-moneyness')
    plt.ylabel('Risk-Neutral Density')
    plt.title('Risk-Neutral Density (Negative values indicate arbitrage)')
    plt.grid(True, alpha=0.3)
    
    # Highlight negative regions if any
    if verification['has_negative']:
        negative_mask = density < 0
        plt.fill_between(k_values, 0, density, where=negative_mask, 
                        color='red', alpha=0.3, label='Arbitrage regions')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    print()


def example_surface_comparison():
    """Example: Compare SVI and SSVI surfaces."""
    print("=== SVI vs SSVI Surface Comparison ===")
    
    # Define grids
    k_values = np.linspace(-1.5, 1.5, 30)
    t_values = np.linspace(0.1, 1.0, 20)
    
    # SVI parameters
    svi_params = {'a': 0.02, 'b': 0.4, 'rho': -0.3, 'm': 0.0, 'sigma': 0.4}
    
    # SSVI parameters
    ssvi_params = {'theta': 0.3, 'phi': 0.5, 'rho': -0.3}
    
    # Compute surfaces
    svi_surface = compute_svi_surface(k_values, t_values, **svi_params)
    ssvi_surface = compute_ssvi_surface(k_values, t_values, **ssvi_params)
    
    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})
    
    K_grid, T_grid = np.meshgrid(k_values, t_values)
    
    # SVI surface
    surf1 = ax1.plot_surface(K_grid, T_grid, svi_surface, cmap='viridis', alpha=0.8)
    ax1.set_title('SVI Surface')
    ax1.set_xlabel('Log-moneyness')
    ax1.set_ylabel('Maturity')
    ax1.set_zlabel('Implied Vol')
    
    # SSVI surface
    surf2 = ax2.plot_surface(K_grid, T_grid, ssvi_surface, cmap='plasma', alpha=0.8)
    ax2.set_title('SSVI Surface')
    ax2.set_xlabel('Log-moneyness')
    ax2.set_ylabel('Maturity')
    ax2.set_zlabel('Implied Vol')
    
    # Difference
    diff = svi_surface - ssvi_surface
    surf3 = ax3.plot_surface(K_grid, T_grid, diff, cmap='RdBu', alpha=0.8)
    ax3.set_title('Difference (SVI - SSVI)')
    ax3.set_xlabel('Log-moneyness')
    ax3.set_ylabel('Maturity')
    ax3.set_zlabel('Vol Difference')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Max absolute difference: {np.max(np.abs(diff)):.6f}")
    print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.6f}")
    print()


def main():
    """Run all examples."""
    print("SVI Models - Refactored Core Examples")
    print("=" * 40)
    
    try:
        example_svi_smile()
        example_density_analysis()
        example_surface_comparison()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
