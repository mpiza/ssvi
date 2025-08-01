#!/usr/bin/env python3
"""
Test script to verify that the analytical derivative of φ(θ) matches 
the numerical derivative within numerical precision.

This test validates the analytical implementation of ∂φ/∂θ using the quotient rule
against numerical differentiation with various step sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parametric_ssvi import compute_phi_rational, compute_phi_rational_derivative

def numerical_phi_derivative(theta: float, p_coeffs, q_coeffs, h=1e-8):
    """Compute numerical derivative for comparison."""
    h_scaled = h * max(theta, 1.0)
    phi_plus = compute_phi_rational(np.array([theta + h_scaled]), p_coeffs, q_coeffs)[0]
    phi_minus = compute_phi_rational(np.array([theta - h_scaled]), p_coeffs, q_coeffs)[0]
    return (phi_plus - phi_minus) / (2 * h_scaled)

def test_phi_derivative_accuracy():
    """Test that analytical and numerical derivatives agree within machine precision."""
    
    # Test parameters
    p_coeffs = [0.1, 0.5, -0.1]  # [p0, p1, p2]
    q_coeffs = [1.0, 0.2, 0.05]  # [q0, q1, q2]
    
    # Test range of theta values
    theta_values = np.linspace(0.01, 2.0, 50)
    
    analytical_derivs = []
    numerical_derivs = []
    relative_errors = []
    
    print("Testing analytical vs numerical φ derivative computation:")
    print("=" * 60)
    
    for theta in theta_values:
        # Analytical derivative
        analytical = compute_phi_rational_derivative(np.array([theta]), p_coeffs, q_coeffs)[0]
        
        # Numerical derivative
        numerical = numerical_phi_derivative(theta, p_coeffs, q_coeffs)
        
        # Relative error
        if abs(analytical) > 1e-12:
            rel_error = abs(analytical - numerical) / abs(analytical)
        else:
            rel_error = abs(analytical - numerical)
        
        analytical_derivs.append(analytical)
        numerical_derivs.append(numerical)
        relative_errors.append(rel_error)
    
    analytical_derivs = np.array(analytical_derivs)
    numerical_derivs = np.array(numerical_derivs)
    relative_errors = np.array(relative_errors)
    
    # Statistics
    max_rel_error = np.max(relative_errors)
    mean_rel_error = np.mean(relative_errors)
    
    print(f"Maximum relative error: {max_rel_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    print(f"Machine epsilon (double): {np.finfo(float).eps:.2e}")
    
    # Check if accuracy is within reasonable bounds
    if max_rel_error < 1e-10:
        print("✅ PASS: Analytical derivative is highly accurate!")
    elif max_rel_error < 1e-6:
        print("⚠️  CAUTION: Analytical derivative has acceptable accuracy")
    else:
        print("❌ FAIL: Analytical derivative has poor accuracy")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot derivatives
    ax1.plot(theta_values, analytical_derivs, 'b-', linewidth=2, label='Analytical ∂φ/∂θ')
    ax1.plot(theta_values, numerical_derivs, 'r--', linewidth=2, label='Numerical ∂φ/∂θ')
    ax1.set_xlabel('θ')
    ax1.set_ylabel('∂φ/∂θ')
    ax1.set_title('Analytical vs Numerical φ Derivative')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot relative errors
    ax2.semilogy(theta_values, relative_errors, 'g-', linewidth=2)
    ax2.axhline(y=np.finfo(float).eps, color='k', linestyle=':', alpha=0.7, 
                label=f'Machine epsilon ({np.finfo(float).eps:.1e})')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Relative Error: |Analytical - Numerical| / |Analytical|')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/phi_derivative_verification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return max_rel_error < 1e-10

def test_expanded_formula():
    """Test the expanded form of the analytical derivative."""
    
    print("\nTesting expanded analytical formula:")
    print("=" * 40)
    
    # Test parameters
    theta = 0.5
    p0, p1, p2 = 0.1, 0.5, -0.1
    q0, q1, q2 = 1.0, 0.2, 0.05
    
    # Method 1: Using the quotient rule function
    p_coeffs = [p0, p1, p2]
    q_coeffs = [q0, q1, q2]
    method1 = compute_phi_rational_derivative(np.array([theta]), p_coeffs, q_coeffs)[0]
    
    # Method 2: Manual expanded calculation
    P_theta = p0 + p1 * theta + p2 * theta**2
    Q_theta = q0 + q1 * theta + q2 * theta**2
    P_prime = p1 + 2 * p2 * theta
    Q_prime = q1 + 2 * q2 * theta
    
    method2 = (P_prime * Q_theta - P_theta * Q_prime) / (Q_theta**2)
    
    print(f"Quotient rule function: {method1:.12f}")
    print(f"Manual calculation:     {method2:.12f}")
    print(f"Difference:             {abs(method1 - method2):.2e}")
    
    if abs(method1 - method2) < 1e-15:
        print("✅ PASS: Both methods agree exactly!")
        return True
    else:
        print("❌ FAIL: Methods disagree!")
        return False

if __name__ == "__main__":
    print("φ(θ) Analytical Derivative Verification")
    print("======================================")
    
    # Test 1: Accuracy vs numerical
    accuracy_test = test_phi_derivative_accuracy()
    
    # Test 2: Expanded formula consistency
    formula_test = test_expanded_formula()
    
    print(f"\nOverall Result:")
    print(f"Accuracy test: {'PASS' if accuracy_test else 'FAIL'}")
    print(f"Formula test:  {'PASS' if formula_test else 'FAIL'}")
    
    if accuracy_test and formula_test:
        print("🎉 All tests passed! Analytical derivative is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
