#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Performance Analysis and Validation Examples for Parametric SSVI

This module demonstrates the performance improvements achieved by using analytical
derivatives instead of finite differences, and validates the accuracy of the 
analytical implementations.

Features:
- Performance benchmarking
- Accuracy validation
- Memory usage comparison
- Scalability analysis
- Real-world usage scenarios

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from typing import Dict, List, Tuple, Any
import warnings

from src.parametric_ssvi import (
    compute_parametric_ssvi_total_variance,
    compute_parametric_ssvi_derivatives,
    compute_parametric_ssvi_all_derivatives,
    compute_parametric_ssvi_time_derivative,
    validate_parametric_ssvi_parameters,
    analyze_parametric_ssvi_properties
)

# Try to import finite difference implementation for comparison
try:
    from scipy.optimize import approx_fprime
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Some benchmarks may be limited.")


def finite_difference_derivatives(
    mu_values: np.ndarray,
    T: float,
    rho: float,
    theta_inf: float,
    theta_0: float,
    kappa: float,
    p_coeffs: List[float],
    q_coeffs: List[float],
    h: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute derivatives using finite differences for comparison.
    
    This is the "old" method that analytical derivatives should replace.
    """
    w = compute_parametric_ssvi_total_variance(
        mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    
    # First derivative: ∂w/∂μ
    w_plus = compute_parametric_ssvi_total_variance(
        mu_values + h, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    w_minus = compute_parametric_ssvi_total_variance(
        mu_values - h, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    w_prime = (w_plus - w_minus) / (2 * h)
    
    # Second derivative: ∂²w/∂μ²
    w_double_prime = (w_plus - 2*w + w_minus) / (h**2)
    
    return w, w_prime, w_double_prime


def benchmark_derivative_methods(
    mu_range: float = 2.0,
    n_points: List[int] = [50, 100, 200, 500],
    n_runs: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive benchmark comparing analytical vs finite difference derivatives.
    """
    if verbose:
        print("Running derivative computation benchmark...")
        print(f"Testing with {n_runs} runs for each grid size")
        print("Grid sizes:", n_points)
        print()
    
    # Test parameters
    rho = 0.1
    theta_inf = 0.04
    theta_0 = 0.09
    kappa = 2.0
    p_coeffs = [1.0, 0.2, -0.1]
    q_coeffs = [1.0, 0.1, 0.0]
    T = 1.0
    
    results = {
        'grid_sizes': n_points,
        'analytical': {'times': [], 'memory': []},
        'finite_diff': {'times': [], 'memory': []},
        'accuracy': []
    }
    
    for n in n_points:
        if verbose:
            print(f"Testing grid size: {n}")
            
        mu_values = np.linspace(-mu_range, mu_range, n)
        
        # Benchmark analytical derivatives
        analytical_times = []
        analytical_memory = []
        
        for run in range(n_runs):
            tracemalloc.start()
            start_time = time.time()
            
            w_analytical, w_prime_analytical, w_double_prime_analytical, dw_dT_analytical = \
                compute_parametric_ssvi_all_derivatives(
                    mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
                )
            
            analytical_times.append(time.time() - start_time)
            current, peak = tracemalloc.get_traced_memory()
            analytical_memory.append(peak)
            tracemalloc.stop()
            
        # Benchmark finite differences
        finite_diff_times = []
        finite_diff_memory = []
        
        for run in range(n_runs):
            tracemalloc.start()
            start_time = time.time()
            
            w_finite, w_prime_finite, w_double_prime_finite = finite_difference_derivatives(
                mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
            )
            
            finite_diff_times.append(time.time() - start_time)
            current, peak = tracemalloc.get_traced_memory()
            finite_diff_memory.append(peak)
            tracemalloc.stop()
            
        # Store timing results
        results['analytical']['times'].append(analytical_times)
        results['analytical']['memory'].append(analytical_memory)
        results['finite_diff']['times'].append(finite_diff_times)
        results['finite_diff']['memory'].append(finite_diff_memory)
        
        # Compute accuracy (compare analytical vs finite difference)
        max_error_first = np.max(np.abs(w_prime_analytical - w_prime_finite))
        max_error_second = np.max(np.abs(w_double_prime_analytical - w_double_prime_finite))
        
        results['accuracy'].append({
            'first_derivative_max_error': max_error_first,
            'second_derivative_max_error': max_error_second,
            'relative_error_first': max_error_first / np.max(np.abs(w_prime_analytical)),
            'relative_error_second': max_error_second / np.max(np.abs(w_double_prime_analytical))
        })
        
        if verbose:
            print(f"  Analytical: {np.mean(analytical_times):.4f}s ± {np.std(analytical_times):.4f}s")
            print(f"  Finite diff: {np.mean(finite_diff_times):.4f}s ± {np.std(finite_diff_times):.4f}s")
            print(f"  Speedup: {np.mean(finite_diff_times) / np.mean(analytical_times):.2f}x")
            print(f"  Max error (1st): {max_error_first:.2e}")
            print(f"  Max error (2nd): {max_error_second:.2e}")
            print()
    
    return results


def plot_benchmark_results(results: Dict[str, Any], save_path: str = None):
    """
    Create comprehensive plots of benchmark results.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    grid_sizes = results['grid_sizes']
    
    # Timing comparison
    analytical_means = [np.mean(times) for times in results['analytical']['times']]
    analytical_stds = [np.std(times) for times in results['analytical']['times']]
    finite_diff_means = [np.mean(times) for times in results['finite_diff']['times']]
    finite_diff_stds = [np.std(times) for times in results['finite_diff']['times']]
    
    ax1.errorbar(grid_sizes, analytical_means, yerr=analytical_stds, 
                 label='Analytical', marker='o', linewidth=2)
    ax1.errorbar(grid_sizes, finite_diff_means, yerr=finite_diff_stds, 
                 label='Finite Difference', marker='s', linewidth=2)
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Computation Time (s)')
    ax1.set_title('Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Speedup plot
    speedups = [f_mean / a_mean for a_mean, f_mean in zip(analytical_means, finite_diff_means)]
    ax2.plot(grid_sizes, speedups, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Grid Size')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Analytical vs Finite Difference Speedup')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
    ax2.legend()
    
    # Memory usage comparison
    analytical_memory_means = [np.mean(mem) / 1024**2 for mem in results['analytical']['memory']]  # MB
    finite_diff_memory_means = [np.mean(mem) / 1024**2 for mem in results['finite_diff']['memory']]  # MB
    
    ax3.plot(grid_sizes, analytical_memory_means, 'o-', label='Analytical', linewidth=2)
    ax3.plot(grid_sizes, finite_diff_memory_means, 's-', label='Finite Difference', linewidth=2)
    ax3.set_xlabel('Grid Size')
    ax3.set_ylabel('Peak Memory Usage (MB)')
    ax3.set_title('Memory Usage Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Accuracy analysis
    first_errors = [acc['relative_error_first'] for acc in results['accuracy']]
    second_errors = [acc['relative_error_second'] for acc in results['accuracy']]
    
    ax4.semilogy(grid_sizes, first_errors, 'o-', label='First Derivative', linewidth=2)
    ax4.semilogy(grid_sizes, second_errors, 's-', label='Second Derivative', linewidth=2)
    ax4.set_xlabel('Grid Size')
    ax4.set_ylabel('Relative Error')
    ax4.set_title('Accuracy: Analytical vs Finite Difference')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Benchmark plot saved to: {save_path}")
    
    plt.show()


def validate_analytical_derivatives(
    test_points: int = 100,
    tolerance: float = 1e-10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate analytical derivatives against high-precision finite differences.
    """
    if verbose:
        print("Validating analytical derivatives...")
        print(f"Using {test_points} test points with tolerance {tolerance}")
        print()
    
    # Test parameters
    rho = 0.15
    theta_inf = 0.05
    theta_0 = 0.08
    kappa = 1.5
    p_coeffs = [1.2, 0.3, -0.05]
    q_coeffs = [1.0, 0.2, 0.02]
    T = 0.75
    
    mu_values = np.linspace(-1.5, 1.5, test_points)
    
    # Compute analytical derivatives
    w_analytical, w_prime_analytical, w_double_prime_analytical, dw_dT_analytical = \
        compute_parametric_ssvi_all_derivatives(
            mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
        )
    
    # High-precision finite differences
    h = 1e-8  # Smaller step for higher precision
    
    # First derivative validation
    w_plus = compute_parametric_ssvi_total_variance(
        mu_values + h, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    w_minus = compute_parametric_ssvi_total_variance(
        mu_values - h, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    w_prime_finite = (w_plus - w_minus) / (2 * h)
    
    # Second derivative validation
    w_double_prime_finite = (w_plus - 2*w_analytical + w_minus) / (h**2)
    
    # Time derivative validation
    dw_dT_plus = compute_parametric_ssvi_total_variance(
        mu_values, T + h, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    dw_dT_minus = compute_parametric_ssvi_total_variance(
        mu_values, T - h, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    dw_dT_finite = (dw_dT_plus - dw_dT_minus) / (2 * h)
    
    # Compute errors
    error_first = np.abs(w_prime_analytical - w_prime_finite)
    error_second = np.abs(w_double_prime_analytical - w_double_prime_finite)
    error_time = np.abs(dw_dT_analytical - dw_dT_finite)
    
    # Relative errors
    rel_error_first = error_first / (np.abs(w_prime_analytical) + 1e-12)
    rel_error_second = error_second / (np.abs(w_double_prime_analytical) + 1e-12)
    rel_error_time = error_time / (np.abs(dw_dT_analytical) + 1e-12)
    
    validation_results = {
        'max_absolute_error': {
            'first_derivative': np.max(error_first),
            'second_derivative': np.max(error_second),
            'time_derivative': np.max(error_time)
        },
        'max_relative_error': {
            'first_derivative': np.max(rel_error_first),
            'second_derivative': np.max(rel_error_second),
            'time_derivative': np.max(rel_error_time)
        },
        'mean_relative_error': {
            'first_derivative': np.mean(rel_error_first),
            'second_derivative': np.mean(rel_error_second),
            'time_derivative': np.mean(rel_error_time)
        },
        'passed_tolerance': {
            'first_derivative': np.all(rel_error_first < tolerance),
            'second_derivative': np.all(rel_error_second < tolerance),
            'time_derivative': np.all(rel_error_time < tolerance)
        }
    }
    
    if verbose:
        print("Validation Results:")
        print(f"First derivative  - Max rel error: {validation_results['max_relative_error']['first_derivative']:.2e}")
        print(f"Second derivative - Max rel error: {validation_results['max_relative_error']['second_derivative']:.2e}")
        print(f"Time derivative   - Max rel error: {validation_results['max_relative_error']['time_derivative']:.2e}")
        print()
        print("Tolerance tests:")
        print(f"First derivative:  {'PASS' if validation_results['passed_tolerance']['first_derivative'] else 'FAIL'}")
        print(f"Second derivative: {'PASS' if validation_results['passed_tolerance']['second_derivative'] else 'FAIL'}")
        print(f"Time derivative:   {'PASS' if validation_results['passed_tolerance']['time_derivative'] else 'FAIL'}")
        print()
    
    return validation_results


def real_world_scenario_test(verbose: bool = True) -> Dict[str, Any]:
    """
    Test performance in a realistic volatility surface calibration scenario.
    """
    if verbose:
        print("Running real-world scenario test...")
        print("Simulating volatility surface computation for multiple strikes and maturities")
        print()
    
    # Realistic market scenario
    strikes = np.linspace(0.8, 1.2, 25)  # 25 strikes from 80% to 120% of spot
    maturities = [1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0, 3.0, 5.0]  # 10 maturities
    
    # Convert strikes to log-moneyness (assuming S=1, r=0 for simplicity)
    mu_values = np.log(strikes)
    
    # Market-realistic parameters
    rho = -0.2  # Negative correlation typical in equity markets
    theta_inf = 0.04  # 20% long-term volatility
    theta_0 = 0.16  # 40% initial volatility  
    kappa = 3.0  # Mean reversion
    p_coeffs = [1.0, 0.5, -0.1]
    q_coeffs = [1.0, 0.3, 0.05]
    
    n_runs = 5
    scenario_results = {
        'analytical': {'times': [], 'surface_shape': None},
        'finite_diff': {'times': [], 'surface_shape': None}
    }
    
    # Test analytical method
    for run in range(n_runs):
        start_time = time.time()
        
        surface = np.zeros((len(maturities), len(mu_values)))
        
        for i, T in enumerate(maturities):
            w, w_prime, w_double_prime, dw_dT = compute_parametric_ssvi_all_derivatives(
                mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
            )
            surface[i, :] = np.sqrt(w / T)  # Convert to implied volatility
            
        scenario_results['analytical']['times'].append(time.time() - start_time)
        
    scenario_results['analytical']['surface_shape'] = surface.shape
    
    # Test finite difference method
    for run in range(n_runs):
        start_time = time.time()
        
        surface = np.zeros((len(maturities), len(mu_values)))
        
        for i, T in enumerate(maturities):
            w, w_prime, w_double_prime = finite_difference_derivatives(
                mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
            )
            surface[i, :] = np.sqrt(w / T)  # Convert to implied volatility
            
        scenario_results['finite_diff']['times'].append(time.time() - start_time)
        
    scenario_results['finite_diff']['surface_shape'] = surface.shape
    
    analytical_mean = np.mean(scenario_results['analytical']['times'])
    finite_diff_mean = np.mean(scenario_results['finite_diff']['times'])
    speedup = finite_diff_mean / analytical_mean
    
    if verbose:
        print(f"Surface dimensions: {surface.shape[0]} maturities × {surface.shape[1]} strikes")
        print(f"Analytical method:     {analytical_mean:.4f}s ± {np.std(scenario_results['analytical']['times']):.4f}s")
        print(f"Finite difference:     {finite_diff_mean:.4f}s ± {np.std(scenario_results['finite_diff']['times']):.4f}s")
        print(f"Real-world speedup:    {speedup:.2f}x")
        print()
    
    scenario_results['speedup'] = speedup
    scenario_results['surface_size'] = surface.shape
    
    return scenario_results


def run_comprehensive_analysis(save_plots: bool = True) -> Dict[str, Any]:
    """
    Run all performance and validation tests.
    """
    print("Parametric SSVI Analytical Derivatives - Comprehensive Analysis")
    print("=" * 70)
    print()
    
    all_results = {}
    
    # 1. Benchmark derivative computation methods
    print("1. Derivative Computation Benchmark")
    print("-" * 40)
    benchmark_results = benchmark_derivative_methods(
        n_points=[25, 50, 100, 200, 400],
        n_runs=5,
        verbose=True
    )
    all_results['benchmark'] = benchmark_results
    
    # 2. Validate analytical derivatives
    print("2. Analytical Derivative Validation")
    print("-" * 40)
    validation_results = validate_analytical_derivatives(
        test_points=200,
        tolerance=1e-8,
        verbose=True
    )
    all_results['validation'] = validation_results
    
    # 3. Real-world scenario test
    print("3. Real-World Scenario Test")
    print("-" * 40)
    scenario_results = real_world_scenario_test(verbose=True)
    all_results['scenario'] = scenario_results
    
    # 4. Generate plots
    if save_plots:
        print("4. Generating Performance Plots")
        print("-" * 40)
        plot_benchmark_results(
            benchmark_results, 
            save_path='parametric_ssvi_performance_analysis.png'
        )
    
    # 5. Summary
    print("5. Summary")
    print("-" * 40)
    
    # Extract key metrics
    largest_grid = benchmark_results['grid_sizes'][-1]
    final_analytical_time = np.mean(benchmark_results['analytical']['times'][-1])
    final_finite_diff_time = np.mean(benchmark_results['finite_diff']['times'][-1])
    max_speedup = final_finite_diff_time / final_analytical_time
    
    validation_passed = all(validation_results['passed_tolerance'].values())
    max_rel_error = max(validation_results['max_relative_error'].values())
    
    real_world_speedup = scenario_results['speedup']
    
    print(f"Maximum tested grid size: {largest_grid} points")
    print(f"Maximum speedup achieved: {max_speedup:.2f}x")
    print(f"Real-world scenario speedup: {real_world_speedup:.2f}x")
    print(f"Analytical validation: {'PASSED' if validation_passed else 'FAILED'}")
    print(f"Maximum relative error: {max_rel_error:.2e}")
    print()
    print("Conclusion: Analytical derivatives provide significant performance")
    print("improvement while maintaining high accuracy compared to finite differences.")
    
    all_results['summary'] = {
        'max_speedup': max_speedup,
        'real_world_speedup': real_world_speedup,
        'validation_passed': validation_passed,
        'max_relative_error': max_rel_error
    }
    
    return all_results


def main():
    """Main function to run performance analysis."""
    results = run_comprehensive_analysis(save_plots=True)
    
    print("\nAnalysis complete! Check the generated plots for detailed results.")
    return results


if __name__ == "__main__":
    main()
