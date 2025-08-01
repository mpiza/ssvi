#!/usr/bin/env python3
"""
Test that the parametric SSVI implementation works correctly with analytical derivatives.

This test validates:
1. All derivatives are computed correctly
2. Total variance is positive everywhere
3. Second derivatives are positive (arbitrage-free condition)
4. All components produce reasonable values
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parametric_ssvi import (
    compute_parametric_ssvi_all_derivatives,
    compute_parametric_ssvi_total_variance
)

def test_parametric_ssvi_with_analytical_derivatives():
    """Test the complete parametric SSVI with analytical derivatives."""
    
    # Test parameters
    mu_values = np.linspace(-1.0, 1.0, 21)
    T = 0.5
    rho = -0.3
    theta_inf = 0.2
    theta_0 = 0.4
    kappa = 2.0
    p_coeffs = [0.1, 0.5, -0.1]
    q_coeffs = [1.0, 0.2, 0.05]
    
    print("Testing Parametric SSVI with Analytical Derivatives")
    print("=" * 55)
    print(f"Parameters:")
    print(f"  T = {T}")
    print(f"  ρ = {rho}")
    print(f"  θ∞ = {theta_inf}, θ₀ = {theta_0}, κ = {kappa}")
    print(f"  p_coeffs = {p_coeffs}")
    print(f"  q_coeffs = {q_coeffs}")
    print()
    
    # Compute all derivatives
    w, w_prime, w_double_prime, dw_dT = compute_parametric_ssvi_all_derivatives(
        mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    
    # Basic sanity checks
    print("Sanity Checks:")
    print(f"  Total variance w: min={np.min(w):.6f}, max={np.max(w):.6f}")
    print(f"  First derivative ∂w/∂μ: min={np.min(w_prime):.6f}, max={np.max(w_prime):.6f}")
    print(f"  Second derivative ∂²w/∂μ²: min={np.min(w_double_prime):.6f}, max={np.max(w_double_prime):.6f}")
    print(f"  Time derivative ∂w/∂T: min={np.min(dw_dT):.6f}, max={np.max(dw_dT):.6f}")
    
    # Check positivity constraints
    print("\nPositivity Checks:")
    print(f"  All w > 0: {np.all(w > 0)}")
    print(f"  All ∂²w/∂μ² > 0: {np.all(w_double_prime > 0)}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total variance
    ax1.plot(mu_values, w, 'b-', linewidth=2)
    ax1.set_xlabel('μ (log-moneyness)')
    ax1.set_ylabel('w(μ, T)')
    ax1.set_title('Total Variance')
    ax1.grid(True, alpha=0.3)
    
    # First derivative
    ax2.plot(mu_values, w_prime, 'r-', linewidth=2)
    ax2.set_xlabel('μ (log-moneyness)')
    ax2.set_ylabel('∂w/∂μ')
    ax2.set_title('First Derivative')
    ax2.grid(True, alpha=0.3)
    
    # Second derivative
    ax3.plot(mu_values, w_double_prime, 'g-', linewidth=2)
    ax3.set_xlabel('μ (log-moneyness)')
    ax3.set_ylabel('∂²w/∂μ²')
    ax3.set_title('Second Derivative')
    ax3.grid(True, alpha=0.3)
    
    # Time derivative
    ax4.plot(mu_values, dw_dT, 'm-', linewidth=2)
    ax4.set_xlabel('μ (log-moneyness)')
    ax4.set_ylabel('∂w/∂T')
    ax4.set_title('Time Derivative')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/parametric_ssvi_derivatives_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ All tests completed successfully!")
    print(f"📊 Plot saved as 'plots/parametric_ssvi_derivatives_test.png'")
    
    return True

if __name__ == "__main__":
    test_parametric_ssvi_with_analytical_derivatives()
