import numpy as np
import matplotlib.pyplot as plt

def old_density_formula(k_values, T, a, b, rho, m, sigma):
    """Old (incorrect) density formula - missing shape function."""
    diff = k_values - m
    sqrt_term = np.sqrt(diff ** 2 + sigma ** 2)
    w = a + b * (rho * diff + sqrt_term)
    w = np.maximum(w, 1e-8)
    
    sqrt_w = np.sqrt(w)
    d2 = -k_values / sqrt_w - sqrt_w / 2
    
    # Old formula (missing shape function g)
    density = np.exp(-d2**2 / 2) / np.sqrt(2 * np.pi * w)
    return density

def correct_density_formula(k_values, T, a, b, rho, m, sigma):
    """Correct density formula with shape function g(y)."""
    y = k_values
    diff = y - m
    sqrt_term = np.sqrt(diff**2 + sigma**2)
    
    # Total variance w(y)
    w = a + b * (rho * diff + sqrt_term)
    w = np.maximum(w, 1e-8)
    
    # First derivative w'(y)
    w_prime = b * (rho + diff / sqrt_term)
    
    # Second derivative w''(y) 
    w_double_prime = b * sigma**2 / (sqrt_term**3)
    
    # Shape function g(y)
    term1 = (1 - y * w_prime / (2 * w))**2
    term2 = (w_prime**2 / 4) * (1/w + 1/4)
    term3 = w_double_prime / 2
    g = term1 - term2 + term3
    
    # d₋ term
    sqrt_w = np.sqrt(w)
    d_minus = -y / sqrt_w - sqrt_w / 2
    
    # Correct density with shape function
    density = (g / np.sqrt(2 * np.pi * w)) * np.exp(-d_minus**2 / 2)
    return density

def compare_density_implementations():
    """Compare old vs correct density implementations."""
    
    # Test parameters
    k_vals = np.linspace(-5, 5, 1000)
    T = 0.5
    a, b, rho, m, sigma = 0.02, 0.4, -0.2, 0.0, 0.4
    
    print("=== COMPARISON: OLD vs CORRECT DENSITY FORMULA ===")
    print(f"SVI parameters: a={a}, b={b}, rho={rho}, m={m}, sigma={sigma}, T={T}")
    
    # Compute both densities
    old_density = old_density_formula(k_vals, T, a, b, rho, m, sigma)
    correct_density = correct_density_formula(k_vals, T, a, b, rho, m, sigma)
    
    # Integration tests
    old_integral = np.trapz(old_density, k_vals)
    correct_integral = np.trapz(correct_density, k_vals)
    
    print(f"\nIntegration results:")
    print(f"Old formula integral:     {old_integral:.6f}")
    print(f"Correct formula integral: {correct_integral:.6f}")
    print(f"Improvement factor:       {old_integral/correct_integral:.3f}")
    
    # Compare at specific points
    print(f"\nPoint-wise comparison at selected k values:")
    test_points = [-2, -1, 0, 1, 2]
    for k_test in test_points:
        idx = np.argmin(np.abs(k_vals - k_test))
        old_val = old_density[idx]
        correct_val = correct_density[idx]
        ratio = old_val / correct_val if correct_val > 0 else np.inf
        print(f"k={k_test:2d}: old={old_val:.6f}, correct={correct_val:.6f}, ratio={ratio:.3f}")
    
    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot both densities
    ax1.plot(k_vals, old_density, 'r-', linewidth=2, label='Old formula (missing g)')
    ax1.plot(k_vals, correct_density, 'b-', linewidth=2, label='Correct formula (with g)')
    ax1.set_xlabel('log-moneyness')
    ax1.set_ylabel('Density')
    ax1.set_title('Density Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 3)
    
    # Plot the ratio
    ratio = old_density / np.maximum(correct_density, 1e-12)
    ax2.plot(k_vals, ratio, 'g-', linewidth=2)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='Ratio = 1')
    ax2.set_xlabel('log-moneyness')
    ax2.set_ylabel('Ratio (old/correct)')
    ax2.set_title('Ratio of Old to Correct Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)
    
    # Plot the shape function g(y)
    y = k_vals
    diff = y - m
    sqrt_term = np.sqrt(diff**2 + sigma**2)
    w = a + b * (rho * diff + sqrt_term)
    w = np.maximum(w, 1e-8)
    w_prime = b * (rho + diff / sqrt_term)
    w_double_prime = b * sigma**2 / (sqrt_term**3)
    
    term1 = (1 - y * w_prime / (2 * w))**2
    term2 = (w_prime**2 / 4) * (1/w + 1/4)
    term3 = w_double_prime / 2
    g = term1 - term2 + term3
    
    ax3.plot(k_vals, g, 'm-', linewidth=2, label='Shape function g(y)')
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='g = 1')
    ax3.set_xlabel('log-moneyness')
    ax3.set_ylabel('g(y)')
    ax3.set_title('Shape Function g(y) - The Missing Factor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-3, 3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nThe shape function g(y) captures the SVI-specific corrections to the")
    print(f"simple Black-Scholes density formula. It accounts for:")
    print(f"1. The effect of volatility derivatives w'(y) and w''(y)")
    print(f"2. The curvature of the SVI volatility surface")
    print(f"3. Proper probability conservation")

if __name__ == "__main__":
    compare_density_implementations()
