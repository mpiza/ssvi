#!/usr/bin/env python3
"""
Parametric SSVI Examples

This script demonstrates the usage of the extended parametric SSVI model with:
- Time-dependent variance level θ(T)
- Rational function φ(θ) for skew dynamics
- Local volatility computation
- Parameter sensitivity analysis
- Comparison with standard SSVI
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from parametric_ssvi import (
    compute_parametric_ssvi_surface,
    compute_parametric_ssvi_volatility_smile,
    compute_parametric_ssvi_derivatives,
    compute_theta_T,
    compute_phi_rational,
    validate_parametric_ssvi_parameters,
    analyze_parametric_ssvi_properties,
    get_default_parametric_ssvi_parameters
)

from svi_models import compute_ssvi_surface
from local_volatility import dupire_local_volatility_from_total_variance


def example_basic_usage():
    """Basic usage example of parametric SSVI."""
    print("=== Basic Parametric SSVI Usage ===")
    
    # Setup grids
    mu_values = np.linspace(-2.0, 2.0, 100)
    T_values = np.linspace(0.1, 3.0, 15)
    T_fixed = 1.0
    
    # Get default parameters
    params = get_default_parametric_ssvi_parameters()
    print(f"Default parameters: {params}")
    
    # Validate parameters
    is_valid, violations = validate_parametric_ssvi_parameters(
        params['rho'], params['theta_inf'], params['theta_0'], 
        params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    print(f"Parameters valid: {is_valid}")
    if violations:
        for violation in violations:
            print(f"  - {violation}")
    
    # Compute volatility smile
    vol_smile = compute_parametric_ssvi_volatility_smile(
        mu_values, T_fixed, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    # Compute surface
    surface = compute_parametric_ssvi_surface(
        mu_values, T_values, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    print(f"Volatility smile range: {np.min(vol_smile):.4f} - {np.max(vol_smile):.4f}")
    print(f"Surface shape: {surface.shape}")
    print(f"Surface variance range: {np.min(surface):.4f} - {np.max(surface):.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot volatility smile
    ax1.plot(mu_values, vol_smile, 'b-', linewidth=2, label=f'Parametric SSVI (T={T_fixed})')
    ax1.set_xlabel('Log-Moneyness μ')
    ax1.set_ylabel('Implied Volatility')
    ax1.set_title('Parametric SSVI Volatility Smile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot surface
    MU, T = np.meshgrid(mu_values, T_values)
    contour = ax2.contour(MU, T, surface, levels=15)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('Log-Moneyness μ')
    ax2.set_ylabel('Time to Maturity T')
    ax2.set_title('Parametric SSVI Total Variance Surface')
    
    plt.tight_layout()
    plt.show()


def example_time_structure_analysis():
    """Analyze time structure properties θ(T) and φ(θ)."""
    print("\n=== Time Structure Analysis ===")
    
    # Parameters
    params = get_default_parametric_ssvi_parameters()
    T_values = np.linspace(0.1, 5.0, 100)
    
    # Compute θ(T)
    theta_T_values = compute_theta_T(T_values, params['theta_inf'], params['theta_0'], params['kappa'])
    
    # Compute φ(θ)
    phi_values = compute_phi_rational(theta_T_values, params['p_coeffs'], params['q_coeffs'])
    
    # Analysis
    print(f"Initial θ(0.1): {theta_T_values[0]:.4f}")
    print(f"Final θ(5.0): {theta_T_values[-1]:.4f}")
    print(f"Asymptotic slope θ∞: {params['theta_inf']:.4f}")
    print(f"φ(θ) range: {np.min(phi_values):.4f} - {np.max(phi_values):.4f}")
    
    # Plot time structure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # θ(T) evolution
    ax1.plot(T_values, theta_T_values, 'b-', linewidth=2, label='θ(T)')
    ax1.plot(T_values, params['theta_inf'] * T_values, 'r--', 
             linewidth=2, label='θ∞T (asymptote)')
    ax1.set_xlabel('Time T')
    ax1.set_ylabel('θ(T)')
    ax1.set_title('Time-Dependent Variance Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # φ(θ) function
    theta_range = np.linspace(0.01, max(theta_T_values) * 1.2, 200)
    phi_range = compute_phi_rational(theta_range, params['p_coeffs'], params['q_coeffs'])
    ax2.plot(theta_range, phi_range, 'g-', linewidth=2, label='φ(θ)')
    ax2.scatter(theta_T_values[::10], phi_values[::10], color='red', s=30, 
               label='φ(θ(T)) for time grid', zorder=5)
    ax2.set_xlabel('θ')
    ax2.set_ylabel('φ(θ)')
    ax2.set_title('Rational Skew Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # φ(θ(T)) evolution over time
    ax3.plot(T_values, phi_values, 'purple', linewidth=2, label='φ(θ(T))')
    ax3.set_xlabel('Time T')
    ax3.set_ylabel('φ(θ(T))')
    ax3.set_title('Time Evolution of Skew Parameter')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ATM total variance term structure
    mu_values = np.linspace(-2, 2, 50)
    atm_idx = len(mu_values) // 2
    surface = compute_parametric_ssvi_surface(
        mu_values, T_values, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    atm_variance = surface[:, atm_idx]
    
    ax4.plot(T_values, atm_variance, 'orange', linewidth=2, label='ATM Total Variance')
    ax4.set_xlabel('Time T')
    ax4.set_ylabel('w(0, T)')
    ax4.set_title('ATM Variance Term Structure')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_local_volatility_analysis():
    """Analyze local volatility from parametric SSVI."""
    print("\n=== Local Volatility Analysis ===")
    
    # Setup
    mu_values = np.linspace(-2.0, 2.0, 100)
    T_fixed = 1.0
    r = 0.02
    
    params = get_default_parametric_ssvi_parameters()
    
    # Compute total variance and derivatives
    w, w_prime, w_double_prime = compute_parametric_ssvi_derivatives(
        mu_values, T_fixed, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    # Compute implied volatility
    implied_vol = np.sqrt(np.maximum(w / T_fixed, 1e-12))
    
    # Compute local volatility (simplified approach: dw_dT = w/T)
    dw_dT = w / T_fixed
    local_vol, is_valid = dupire_local_volatility_from_total_variance(
        mu_values, T_fixed, w, w_prime, w_double_prime, dw_dT=dw_dT, r=r
    )
    
    # Statistics
    valid_count = np.sum(is_valid)
    print(f"Valid local volatility points: {valid_count}/{len(mu_values)}")
    print(f"Implied volatility range: {np.min(implied_vol):.4f} - {np.max(implied_vol):.4f}")
    
    if valid_count > 0:
        valid_lv = local_vol[is_valid]
        print(f"Local volatility range: {np.min(valid_lv):.4f} - {np.max(valid_lv):.4f}")
        
        # Ratio analysis
        valid_iv = implied_vol[is_valid]
        ratio = valid_lv / valid_iv
        print(f"LV/IV ratio range: {np.min(ratio):.4f} - {np.max(ratio):.4f}")
    
    # Plot analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Implied vs Local volatility
    local_vol_masked = np.where(is_valid, local_vol, np.nan)
    ax1.plot(mu_values, implied_vol, 'b-', linewidth=2, label='Implied Volatility', alpha=0.8)
    ax1.plot(mu_values, local_vol_masked, 'r-', linewidth=2, label='Local Volatility', alpha=0.8)
    
    if np.any(~is_valid):
        ax1.scatter(mu_values[~is_valid], np.full(np.sum(~is_valid), np.nan),
                   color='red', s=20, marker='x', label='Invalid LV', zorder=5)
    
    ax1.set_xlabel('Log-Moneyness μ')
    ax1.set_ylabel('Volatility')
    ax1.set_title(f'Implied vs Local Volatility (T={T_fixed})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volatility ratio
    if valid_count > 0:
        ax2.plot(mu_values[is_valid], ratio, 'g-', linewidth=2, label='LV/IV Ratio')
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='LV = IV')
        ax2.set_xlabel('Log-Moneyness μ')
        ax2.set_ylabel('Local/Implied Ratio')
        ax2.set_title('Volatility Ratio Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Total variance and derivatives
    ax3.plot(mu_values, w, 'b-', linewidth=2, label='w(μ,T)')
    ax3.plot(mu_values, w_prime, 'g-', linewidth=2, label="w'(μ,T)")
    ax3.plot(mu_values, w_double_prime, 'r-', linewidth=2, label='w"(μ,T)')
    ax3.set_xlabel('Log-Moneyness μ')
    ax3.set_ylabel('Derivatives')
    ax3.set_title('Total Variance and Derivatives')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Validity analysis
    validity_int = is_valid.astype(int)
    ax4.plot(mu_values, validity_int, 'k-', linewidth=2, label='Validity (1=Valid, 0=Invalid)')
    ax4.fill_between(mu_values, 0, validity_int, alpha=0.3, color='green')
    ax4.set_xlabel('Log-Moneyness μ')
    ax4.set_ylabel('Validity')
    ax4.set_title('Local Volatility Validity Regions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()


def example_parameter_sensitivity():
    """Analyze sensitivity to different parameters."""
    print("\n=== Parameter Sensitivity Analysis ===")
    
    # Base parameters
    base_params = get_default_parametric_ssvi_parameters()
    mu_values = np.linspace(-2.0, 2.0, 50)
    T_fixed = 1.0
    
    # Base volatility smile
    base_vol = compute_parametric_ssvi_volatility_smile(
        mu_values, T_fixed, base_params['rho'], base_params['theta_inf'], 
        base_params['theta_0'], base_params['kappa'], 
        base_params['p_coeffs'], base_params['q_coeffs']
    )
    
    # Parameter variations
    sensitivities = {}
    
    # ρ sensitivity
    rho_values = [-0.7, -0.5, -0.3, -0.1, 0.1]
    sensitivities['rho'] = []
    for rho in rho_values:
        vol = compute_parametric_ssvi_volatility_smile(
            mu_values, T_fixed, rho, base_params['theta_inf'], 
            base_params['theta_0'], base_params['kappa'], 
            base_params['p_coeffs'], base_params['q_coeffs']
        )
        sensitivities['rho'].append((rho, vol))
    
    # κ sensitivity
    kappa_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    sensitivities['kappa'] = []
    for kappa in kappa_values:
        vol = compute_parametric_ssvi_volatility_smile(
            mu_values, T_fixed, base_params['rho'], base_params['theta_inf'], 
            base_params['theta_0'], kappa, 
            base_params['p_coeffs'], base_params['q_coeffs']
        )
        sensitivities['kappa'].append((kappa, vol))
    
    # θ∞ sensitivity
    theta_inf_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    sensitivities['theta_inf'] = []
    for theta_inf in theta_inf_values:
        vol = compute_parametric_ssvi_volatility_smile(
            mu_values, T_fixed, base_params['rho'], theta_inf, 
            base_params['theta_0'], base_params['kappa'], 
            base_params['p_coeffs'], base_params['q_coeffs']
        )
        sensitivities['theta_inf'].append((theta_inf, vol))
    
    # Plot sensitivity analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Base case
    ax1.plot(mu_values, base_vol, 'k-', linewidth=3, label='Base Case', alpha=0.8)
    ax1.set_xlabel('Log-Moneyness μ')
    ax1.set_ylabel('Implied Volatility')
    ax1.set_title('Base Parametric SSVI Smile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ρ sensitivity
    for rho, vol in sensitivities['rho']:
        ax2.plot(mu_values, vol, linewidth=2, label=f'ρ = {rho:.1f}', alpha=0.8)
    ax2.set_xlabel('Log-Moneyness μ')
    ax2.set_ylabel('Implied Volatility')
    ax2.set_title('Sensitivity to Correlation ρ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # κ sensitivity
    for kappa, vol in sensitivities['kappa']:
        ax3.plot(mu_values, vol, linewidth=2, label=f'κ = {kappa:.1f}', alpha=0.8)
    ax3.set_xlabel('Log-Moneyness μ')
    ax3.set_ylabel('Implied Volatility')
    ax3.set_title('Sensitivity to Mean Reversion κ')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # θ∞ sensitivity
    for theta_inf, vol in sensitivities['theta_inf']:
        ax4.plot(mu_values, vol, linewidth=2, label=f'θ∞ = {theta_inf:.2f}', alpha=0.8)
    ax4.set_xlabel('Log-Moneyness μ')
    ax4.set_ylabel('Implied Volatility')
    ax4.set_title('Sensitivity to Long-term Variance θ∞')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print sensitivity statistics
    print("Sensitivity Analysis Results:")
    for param_name, param_data in sensitivities.items():
        atm_idx = len(mu_values) // 2
        atm_vols = [vol[atm_idx] for _, vol in param_data]
        print(f"{param_name}: ATM vol range = {np.min(atm_vols):.4f} - {np.max(atm_vols):.4f}")


def example_comparison_with_standard_ssvi():
    """Compare parametric SSVI with standard SSVI."""
    print("\n=== Comparison with Standard SSVI ===")
    
    # Setup
    mu_values = np.linspace(-2.0, 2.0, 50)
    T_values = np.linspace(0.1, 3.0, 10)
    
    # Parametric SSVI parameters
    params = get_default_parametric_ssvi_parameters()
    
    # Standard SSVI parameters (approximate mapping)
    # For comparison, we'll use fixed θ and φ values
    theta_standard = 0.2  # Fixed variance level
    phi_standard = 1.0    # Fixed skew parameter
    rho_standard = params['rho']
    
    # Compute surfaces
    parametric_surface = compute_parametric_ssvi_surface(
        mu_values, T_values, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    standard_surface = compute_ssvi_surface(
        mu_values, T_values, theta_standard, phi_standard, rho_standard
    )
    
    # Analysis
    diff_surface = parametric_surface - standard_surface
    max_abs_diff = np.max(np.abs(diff_surface))
    rms_diff = np.sqrt(np.mean(diff_surface**2))
    
    print(f"Maximum absolute difference: {max_abs_diff:.6f}")
    print(f"RMS difference: {rms_diff:.6f}")
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    MU, T = np.meshgrid(mu_values, T_values)
    
    # Parametric SSVI
    contour1 = ax1.contour(MU, T, parametric_surface, levels=12)
    ax1.clabel(contour1, inline=True, fontsize=8)
    ax1.set_xlabel('Log-Moneyness μ')
    ax1.set_ylabel('Time to Maturity T')
    ax1.set_title('Parametric SSVI Surface')
    
    # Standard SSVI
    contour2 = ax2.contour(MU, T, standard_surface, levels=12)
    ax2.clabel(contour2, inline=True, fontsize=8)
    ax2.set_xlabel('Log-Moneyness μ')
    ax2.set_ylabel('Time to Maturity T')
    ax2.set_title('Standard SSVI Surface')
    
    # Difference
    contour3 = ax3.contour(MU, T, diff_surface, levels=15)
    ax3.clabel(contour3, inline=True, fontsize=8)
    ax3.set_xlabel('Log-Moneyness μ')
    ax3.set_ylabel('Time to Maturity T')
    ax3.set_title('Difference (Parametric - Standard)')
    
    # ATM term structure comparison
    atm_idx = len(mu_values) // 2
    ax4.plot(T_values, parametric_surface[:, atm_idx], 'b-', linewidth=2, 
             label='Parametric SSVI', alpha=0.8)
    ax4.plot(T_values, standard_surface[:, atm_idx], 'r--', linewidth=2, 
             label='Standard SSVI', alpha=0.8)
    ax4.set_xlabel('Time to Maturity T')
    ax4.set_ylabel('ATM Total Variance')
    ax4.set_title('ATM Term Structure Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all parametric SSVI examples."""
    print("Parametric SSVI Model Examples")
    print("===============================")
    print("Demonstrating the extended SSVI model with parametric time dependence:")
    print("- Time-dependent variance level θ(T)")
    print("- Rational function φ(θ) for skew dynamics")
    print("- Local volatility computation")
    print("- Parameter sensitivity analysis")
    print("- Comparison with standard SSVI")
    
    try:
        example_basic_usage()
        example_time_structure_analysis()
        example_local_volatility_analysis()
        example_parameter_sensitivity()
        example_comparison_with_standard_ssvi()
        
        print("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
