#!/usr/bin/env python3
"""
Test Suite for Parametric SSVI Model

This test suite validates the extended parametric SSVI implementation:
- Mathematical correctness of formulas
- Parameter validation
- Edge cases and numerical stability
- Comparison with analytical results where possible
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.parametric_ssvi import (
    compute_theta_T,
    compute_phi_rational,
    compute_parametric_ssvi_total_variance,
    compute_parametric_ssvi_volatility_smile,
    compute_parametric_ssvi_derivatives,
    validate_parametric_ssvi_parameters,
    get_default_parametric_ssvi_parameters
)


def test_theta_T_function():
    """Test the time-dependent variance level θ(T)."""
    print("=== Testing θ(T) Function ===")
    
    # Test parameters
    theta_inf = 0.2
    theta_0 = 0.1
    kappa = 2.0
    T_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    # Compute θ(T)
    theta_T = compute_theta_T(T_values, theta_inf, theta_0, kappa)
    
    print(f"θ∞ = {theta_inf}, θ0 = {theta_0}, κ = {kappa}")
    print(f"T values: {T_values}")
    print(f"θ(T) values: {theta_T}")
    
    # Test asymptotic behavior
    large_T = 100.0
    theta_large = compute_theta_T(np.array([large_T]), theta_inf, theta_0, kappa)[0]
    expected_large = theta_inf * large_T
    asymptotic_error = abs(theta_large - expected_large) / expected_large
    
    print(f"Asymptotic test: θ({large_T}) = {theta_large:.6f}, θ∞T = {expected_large:.6f}")
    print(f"Relative error: {asymptotic_error:.8f}")
    
    # Test initial behavior (T → 0)
    small_T = 0.001
    theta_small = compute_theta_T(np.array([small_T]), theta_inf, theta_0, kappa)[0]
    expected_small = theta_0 * small_T
    initial_error = abs(theta_small - expected_small) / expected_small
    
    print(f"Initial test: θ({small_T}) = {theta_small:.8f}, θ0*T = {expected_small:.8f}")
    print(f"Relative error: {initial_error:.8f}")
    
    # Test monotonicity (should be increasing)
    is_monotonic = np.all(np.diff(theta_T) > 0)
    print(f"Monotonicity check: {'PASSED' if is_monotonic else 'FAILED'}")
    
    # Test κ → 0 limit (linear growth)
    kappa_small = 1e-10
    theta_linear = compute_theta_T(T_values, theta_inf, theta_0, kappa_small)
    expected_linear = theta_0 * T_values
    linear_error = np.max(np.abs(theta_linear - expected_linear) / expected_linear)
    
    print(f"Linear limit test (κ→0): max relative error = {linear_error:.8f}")
    
    success = (asymptotic_error < 1e-6 and initial_error < 1e-6 and 
               is_monotonic and linear_error < 1e-6)
    print(f"θ(T) test: {'PASSED' if success else 'FAILED'}")
    
    return success


def test_phi_rational_function():
    """Test the rational function φ(θ)."""
    print("\n=== Testing φ(θ) Rational Function ===")
    
    # Test parameters
    p_coeffs = [0.5, 0.1, 0.02]  # p0 + p1*θ + p2*θ²
    q_coeffs = [1.0, 0.05, 0.01] # q0 + q1*θ + q2*θ²
    theta_values = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    
    # Compute φ(θ)
    phi_values = compute_phi_rational(theta_values, p_coeffs, q_coeffs)
    
    print(f"p coefficients: {p_coeffs}")
    print(f"q coefficients: {q_coeffs}")
    print(f"θ values: {theta_values}")
    print(f"φ(θ) values: {phi_values}")
    
    # Test positivity
    all_positive = np.all(phi_values > 0)
    print(f"Positivity check: {'PASSED' if all_positive else 'FAILED'}")
    
    # Test continuity (no jumps)
    theta_fine = np.linspace(0.01, 5.0, 1000)
    phi_fine = compute_phi_rational(theta_fine, p_coeffs, q_coeffs)
    max_diff = np.max(np.abs(np.diff(phi_fine)))
    continuity_ok = max_diff < 1.0  # Reasonable continuity
    
    print(f"Continuity check: max difference = {max_diff:.6f}, {'PASSED' if continuity_ok else 'FAILED'}")
    
    # Test special case: linear function (p2=0, q2=0)
    p_linear = [0.5, 0.1, 0.0]
    q_linear = [1.0, 0.05, 0.0]
    phi_linear = compute_phi_rational(theta_values, p_linear, q_linear)
    expected_linear = (p_linear[0] + p_linear[1] * theta_values) / (q_linear[0] + q_linear[1] * theta_values)
    linear_error = np.max(np.abs(phi_linear - expected_linear) / expected_linear)
    
    print(f"Linear case test: max relative error = {linear_error:.8f}")
    
    success = all_positive and continuity_ok and linear_error < 1e-12
    print(f"φ(θ) test: {'PASSED' if success else 'FAILED'}")
    
    return success


def test_parametric_ssvi_formula():
    """Test the parametric SSVI total variance formula."""
    print("\n=== Testing Parametric SSVI Formula ===")
    
    # Test parameters
    params = get_default_parametric_ssvi_parameters()
    mu_values = np.linspace(-2.0, 2.0, 21)
    T = 1.0
    
    # Compute total variance
    w = compute_parametric_ssvi_total_variance(
        mu_values, T, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    print(f"Test parameters: {params}")
    print(f"T = {T}, μ range: [{mu_values[0]:.1f}, {mu_values[-1]:.1f}]")
    print(f"Total variance range: [{np.min(w):.6f}, {np.max(w):.6f}]")
    
    # Test positivity
    all_positive = np.all(w > 0)
    print(f"Positivity check: {'PASSED' if all_positive else 'FAILED'}")
    
    # Test ATM value
    atm_idx = len(mu_values) // 2
    w_atm = w[atm_idx]
    print(f"ATM total variance: {w_atm:.6f}")
    
    # Test symmetry for ρ = 0 case
    rho_zero = 0.0
    w_symmetric = compute_parametric_ssvi_total_variance(
        mu_values, T, rho_zero, params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    # Check if symmetric around μ = 0
    left_half = w_symmetric[:len(w_symmetric)//2]
    right_half = w_symmetric[len(w_symmetric)//2+1:][::-1]  # Reverse right half
    symmetry_error = np.max(np.abs(left_half - right_half)) if len(left_half) == len(right_half) else 0
    
    print(f"Symmetry test (ρ=0): max error = {symmetry_error:.8f}")
    
    # Test monotonicity in T
    T_values = np.array([0.1, 0.5, 1.0, 2.0])
    mu_test = 0.0
    w_term_structure = []
    for T_test in T_values:
        w_t = compute_parametric_ssvi_total_variance(
            np.array([mu_test]), T_test, params['rho'], params['theta_inf'], 
            params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
        )[0]
        w_term_structure.append(w_t)
    
    w_term_structure = np.array(w_term_structure)
    is_increasing = np.all(np.diff(w_term_structure) > 0)
    print(f"Term structure monotonicity: {'PASSED' if is_increasing else 'FAILED'}")
    print(f"w(0,T) for T={T_values}: {w_term_structure}")
    
    success = all_positive and symmetry_error < 1e-10 and is_increasing
    print(f"Parametric SSVI formula test: {'PASSED' if success else 'FAILED'}")
    
    return success


def test_derivatives():
    """Test derivative computations."""
    print("\n=== Testing Derivative Computations ===")
    
    # Test parameters
    params = get_default_parametric_ssvi_parameters()
    mu_values = np.linspace(-1.0, 1.0, 11)
    T = 1.0
    
    # Compute derivatives
    w, w_prime, w_double_prime = compute_parametric_ssvi_derivatives(
        mu_values, T, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    print(f"μ range: [{mu_values[0]:.1f}, {mu_values[-1]:.1f}]")
    print(f"w range: [{np.min(w):.6f}, {np.max(w):.6f}]")
    print(f"w' range: [{np.min(w_prime):.6f}, {np.max(w_prime):.6f}]")
    print(f"w'' range: [{np.min(w_double_prime):.6f}, {np.max(w_double_prime):.6f}]")
    
    # Test derivative accuracy using central differences
    h = 1e-6
    w_plus = compute_parametric_ssvi_total_variance(
        mu_values + h, T, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    w_minus = compute_parametric_ssvi_total_variance(
        mu_values - h, T, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    # Numerical first derivative
    w_prime_numerical = (w_plus - w_minus) / (2 * h)
    first_derivative_error = np.max(np.abs(w_prime - w_prime_numerical) / np.abs(w_prime_numerical))
    
    print(f"First derivative error: {first_derivative_error:.8f}")
    
    # Numerical second derivative
    w_double_prime_numerical = (w_plus - 2*w + w_minus) / (h**2)
    second_derivative_error = np.max(np.abs(w_double_prime - w_double_prime_numerical) / 
                                   np.abs(w_double_prime_numerical))
    
    print(f"Second derivative error: {second_derivative_error:.8f}")
    
    success = first_derivative_error < 1e-4 and second_derivative_error < 1e-3
    print(f"Derivative test: {'PASSED' if success else 'FAILED'}")
    
    return success


def test_parameter_validation():
    """Test parameter validation."""
    print("\n=== Testing Parameter Validation ===")
    
    # Valid parameters
    valid_params = get_default_parametric_ssvi_parameters()
    is_valid, violations = validate_parametric_ssvi_parameters(
        valid_params['rho'], valid_params['theta_inf'], valid_params['theta_0'],
        valid_params['kappa'], valid_params['p_coeffs'], valid_params['q_coeffs']
    )
    
    print(f"Default parameters validation: {'PASSED' if is_valid else 'FAILED'}")
    if violations:
        for violation in violations:
            print(f"  - {violation}")
    
    # Test invalid ρ
    is_valid_rho, violations_rho = validate_parametric_ssvi_parameters(
        1.5, valid_params['theta_inf'], valid_params['theta_0'],
        valid_params['kappa'], valid_params['p_coeffs'], valid_params['q_coeffs']
    )
    print(f"Invalid ρ test: {'PASSED' if not is_valid_rho else 'FAILED'}")
    
    # Test negative θ∞
    is_valid_theta, violations_theta = validate_parametric_ssvi_parameters(
        valid_params['rho'], -0.1, valid_params['theta_0'],
        valid_params['kappa'], valid_params['p_coeffs'], valid_params['q_coeffs']
    )
    print(f"Negative θ∞ test: {'PASSED' if not is_valid_theta else 'FAILED'}")
    
    # Test negative κ
    is_valid_kappa, violations_kappa = validate_parametric_ssvi_parameters(
        valid_params['rho'], valid_params['theta_inf'], valid_params['theta_0'],
        -1.0, valid_params['p_coeffs'], valid_params['q_coeffs']
    )
    print(f"Negative κ test: {'PASSED' if not is_valid_kappa else 'FAILED'}")
    
    success = is_valid and not is_valid_rho and not is_valid_theta and not is_valid_kappa
    print(f"Parameter validation test: {'PASSED' if success else 'FAILED'}")
    
    return success


def test_edge_cases():
    """Test edge cases and numerical stability."""
    print("\n=== Testing Edge Cases ===")
    
    params = get_default_parametric_ssvi_parameters()
    
    # Test very small time
    T_small = 1e-6
    mu_values = np.array([-1.0, 0.0, 1.0])
    
    try:
        w_small = compute_parametric_ssvi_total_variance(
            mu_values, T_small, params['rho'], params['theta_inf'], 
            params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
        )
        small_time_ok = np.all(np.isfinite(w_small)) and np.all(w_small > 0)
        print(f"Small time test (T={T_small}): {'PASSED' if small_time_ok else 'FAILED'}")
    except Exception as e:
        print(f"Small time test: FAILED ({str(e)})")
        small_time_ok = False
    
    # Test very large time
    T_large = 100.0
    
    try:
        w_large = compute_parametric_ssvi_total_variance(
            mu_values, T_large, params['rho'], params['theta_inf'], 
            params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
        )
        large_time_ok = np.all(np.isfinite(w_large)) and np.all(w_large > 0)
        print(f"Large time test (T={T_large}): {'PASSED' if large_time_ok else 'FAILED'}")
    except Exception as e:
        print(f"Large time test: FAILED ({str(e)})")
        large_time_ok = False
    
    # Test extreme moneyness
    mu_extreme = np.array([-10.0, 10.0])
    T_test = 1.0
    
    try:
        w_extreme = compute_parametric_ssvi_total_variance(
            mu_extreme, T_test, params['rho'], params['theta_inf'], 
            params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
        )
        extreme_moneyness_ok = np.all(np.isfinite(w_extreme)) and np.all(w_extreme > 0)
        print(f"Extreme moneyness test: {'PASSED' if extreme_moneyness_ok else 'FAILED'}")
    except Exception as e:
        print(f"Extreme moneyness test: FAILED ({str(e)})")
        extreme_moneyness_ok = False
    
    # Test κ → 0 (no mean reversion)
    kappa_zero = 1e-15
    
    try:
        w_no_mr = compute_parametric_ssvi_total_variance(
            mu_values, T_test, params['rho'], params['theta_inf'], 
            params['theta_0'], kappa_zero, params['p_coeffs'], params['q_coeffs']
        )
        no_mean_reversion_ok = np.all(np.isfinite(w_no_mr)) and np.all(w_no_mr > 0)
        print(f"No mean reversion test (κ≈0): {'PASSED' if no_mean_reversion_ok else 'FAILED'}")
    except Exception as e:
        print(f"No mean reversion test: FAILED ({str(e)})")
        no_mean_reversion_ok = False
    
    success = small_time_ok and large_time_ok and extreme_moneyness_ok and no_mean_reversion_ok
    print(f"Edge cases test: {'PASSED' if success else 'FAILED'}")
    
    return success


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("Parametric SSVI Model Test Suite")
    print("=================================")
    
    tests = [
        ("θ(T) Function", test_theta_T_function),
        ("φ(θ) Rational Function", test_phi_rational_function),
        ("Parametric SSVI Formula", test_parametric_ssvi_formula),
        ("Derivative Computations", test_derivatives),
        ("Parameter Validation", test_parameter_validation),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"ERROR in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    print("\n=== Test Results Summary ===")
    passed = 0
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests PASSED! Parametric SSVI implementation is validated.")
    else:
        print("⚠️  Some tests FAILED. Please review the implementation.")
    
    return passed == len(results)


def create_validation_plots():
    """Create validation plots for visual inspection."""
    print("\n=== Creating Validation Plots ===")
    
    params = get_default_parametric_ssvi_parameters()
    
    # Setup grids
    mu_values = np.linspace(-3.0, 3.0, 100)
    T_values = np.linspace(0.1, 5.0, 50)
    T_fixed = 1.0
    
    # Create comprehensive validation plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. θ(T) and φ(θ) evolution
    theta_T_values = compute_theta_T(T_values, params['theta_inf'], params['theta_0'], params['kappa'])
    phi_values = compute_phi_rational(theta_T_values, params['p_coeffs'], params['q_coeffs'])
    
    ax1.plot(T_values, theta_T_values, 'b-', linewidth=2, label='θ(T)')
    ax1.plot(T_values, params['theta_inf'] * T_values, 'r--', linewidth=2, label='θ∞T (asymptote)')
    ax1.set_xlabel('Time T')
    ax1.set_ylabel('θ(T)')
    ax1.set_title('Time-Dependent Variance Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Volatility smile
    vol_smile = compute_parametric_ssvi_volatility_smile(
        mu_values, T_fixed, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    ax2.plot(mu_values, vol_smile, 'g-', linewidth=2, label=f'Parametric SSVI (T={T_fixed})')
    ax2.set_xlabel('Log-Moneyness μ')
    ax2.set_ylabel('Implied Volatility')
    ax2.set_title('Parametric SSVI Volatility Smile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Surface contour
    surface = compute_parametric_ssvi_surface(
        mu_values, T_values, params['rho'], params['theta_inf'], 
        params['theta_0'], params['kappa'], params['p_coeffs'], params['q_coeffs']
    )
    
    MU, T = np.meshgrid(mu_values, T_values)
    contour = ax3.contour(MU, T, surface, levels=15)
    ax3.clabel(contour, inline=True, fontsize=8)
    ax3.set_xlabel('Log-Moneyness μ')
    ax3.set_ylabel('Time to Maturity T')
    ax3.set_title('Parametric SSVI Total Variance Surface')
    
    # 4. φ(θ) function
    theta_range = np.linspace(0.01, max(theta_T_values) * 1.2, 200)
    phi_range = compute_phi_rational(theta_range, params['p_coeffs'], params['q_coeffs'])
    
    ax4.plot(theta_range, phi_range, 'purple', linewidth=2, label='φ(θ)')
    ax4.scatter(theta_T_values[::5], phi_values[::5], color='red', s=30, 
               label='φ(θ(T)) for time grid', zorder=5)
    ax4.set_xlabel('θ')
    ax4.set_ylabel('φ(θ)')
    ax4.set_title('Rational Skew Function')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Parametric SSVI Model Validation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("Validation plots created successfully!")


def main():
    """Main test runner."""
    print("Starting Parametric SSVI Test Suite...")
    
    # Run comprehensive tests
    all_passed = run_comprehensive_test()
    
    # Create validation plots
    create_validation_plots()
    
    if all_passed:
        print("\n✅ Parametric SSVI implementation is fully validated!")
        print("The extended SSVI model with parametric time dependence is ready for use.")
    else:
        print("\n❌ Some tests failed. Please review the implementation.")
    
    return all_passed


if __name__ == "__main__":
    main()
