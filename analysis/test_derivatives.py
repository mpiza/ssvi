#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Debug and test analytical derivatives for parametric SSVI.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.parametric_ssvi import (
    compute_parametric_ssvi_total_variance,
    compute_parametric_ssvi_derivatives,
    compute_parametric_ssvi_all_derivatives
)

def test_analytical_derivatives():
    """Simple test of analytical derivatives."""
    print("Testing analytical derivatives...")
    
    # Simple test case
    mu = np.array([0.0])  # At-the-money
    T = 1.0
    rho = 0.1
    theta_inf = 0.04
    theta_0 = 0.09
    kappa = 2.0
    p_coeffs = [1.0, 0.2, -0.1]
    q_coeffs = [1.0, 0.1, 0.0]
    
    # Compute analytical derivatives
    w_analytical, w_prime_analytical, w_double_prime_analytical = \
        compute_parametric_ssvi_derivatives(mu, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs)
    
    print(f"At μ=0, T=1:")
    print(f"w = {w_analytical[0]:.6f}")
    print(f"∂w/∂μ = {w_prime_analytical[0]:.6f}")
    print(f"∂²w/∂μ² = {w_double_prime_analytical[0]:.6f}")
    
    # Test finite differences
    h = 1e-6
    
    # First derivative
    w_plus = compute_parametric_ssvi_total_variance(mu + h, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs)
    w_minus = compute_parametric_ssvi_total_variance(mu - h, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs)
    w_prime_finite = (w_plus - w_minus) / (2 * h)
    
    # Second derivative
    w_double_prime_finite = (w_plus - 2*w_analytical + w_minus) / (h**2)
    
    print(f"\nFinite differences (h={h}):")
    print(f"∂w/∂μ = {w_prime_finite[0]:.6f}")
    print(f"∂²w/∂μ² = {w_double_prime_finite[0]:.6f}")
    
    print(f"\nErrors:")
    print(f"First derivative error: {abs(w_prime_analytical[0] - w_prime_finite[0]):.2e}")
    print(f"Second derivative error: {abs(w_double_prime_analytical[0] - w_double_prime_finite[0]):.2e}")
    
    # Plot comparison over range
    mu_range = np.linspace(-1.0, 1.0, 100)
    
    w_analytical_range, w_prime_analytical_range, w_double_prime_analytical_range = \
        compute_parametric_ssvi_derivatives(mu_range, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs)
    
    # Finite differences for comparison
    w_prime_finite_range = np.zeros_like(mu_range)
    w_double_prime_finite_range = np.zeros_like(mu_range)
    
    for i, mu_val in enumerate(mu_range):
        w_plus = compute_parametric_ssvi_total_variance(np.array([mu_val + h]), T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs)
        w_minus = compute_parametric_ssvi_total_variance(np.array([mu_val - h]), T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs)
        w_center = compute_parametric_ssvi_total_variance(np.array([mu_val]), T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs)
        
        w_prime_finite_range[i] = (w_plus[0] - w_minus[0]) / (2 * h)
        w_double_prime_finite_range[i] = (w_plus[0] - 2*w_center[0] + w_minus[0]) / (h**2)
    
    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Total variance
    ax1.plot(mu_range, w_analytical_range, 'b-', label='Total Variance w(μ)', linewidth=2)
    ax1.set_xlabel('μ')
    ax1.set_ylabel('w(μ)')
    ax1.set_title('Parametric SSVI Total Variance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # First derivative comparison
    ax2.plot(mu_range, w_prime_analytical_range, 'b-', label='Analytical ∂w/∂μ', linewidth=2)
    ax2.plot(mu_range, w_prime_finite_range, 'r--', label='Finite Difference ∂w/∂μ', linewidth=1, alpha=0.7)
    ax2.set_xlabel('μ')
    ax2.set_ylabel('∂w/∂μ')
    ax2.set_title('First Derivative Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Second derivative comparison
    ax3.plot(mu_range, w_double_prime_analytical_range, 'b-', label='Analytical ∂²w/∂μ²', linewidth=2)
    ax3.plot(mu_range, w_double_prime_finite_range, 'r--', label='Finite Difference ∂²w/∂μ²', linewidth=1, alpha=0.7)
    ax3.set_xlabel('μ')
    ax3.set_ylabel('∂²w/∂μ²')
    ax3.set_title('Second Derivative Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('derivative_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error analysis
    error_first = np.abs(w_prime_analytical_range - w_prime_finite_range)
    error_second = np.abs(w_double_prime_analytical_range - w_double_prime_finite_range)
    
    print(f"\nError Analysis over range:")
    print(f"Max first derivative error: {np.max(error_first):.2e}")
    print(f"Max second derivative error: {np.max(error_second):.2e}")
    print(f"Mean first derivative error: {np.mean(error_first):.2e}")
    print(f"Mean second derivative error: {np.mean(error_second):.2e}")

if __name__ == "__main__":
    test_analytical_derivatives()
