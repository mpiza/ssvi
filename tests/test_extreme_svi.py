import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

def compute_svi_total_variance(k_values, a, b, rho, m, sigma):
    """Compute SVI total variance without any clipping."""
    diff = k_values - m
    sqrt_term = np.sqrt(diff ** 2 + sigma ** 2)
    w = a + b * (rho * diff + sqrt_term)
    return w

def compute_risk_neutral_density_unclipped(k_values, T, a, b, rho, m, sigma):
    """Compute risk-neutral density allowing negative values to show arbitrage."""
    # Compute total variance (don't clip!)
    w = compute_svi_total_variance(k_values, a, b, rho, m, sigma)
    
    # For numerical computation, we need w > 0 for sqrt and log
    # But let's see what happens when we push extreme parameters
    w_safe = np.maximum(w, 1e-12)  # Minimal safety for numerical computation
    
    # Compute d2 from Black-Scholes
    sqrt_wT = np.sqrt(w_safe * T)
    d2 = -k_values / sqrt_wT - sqrt_wT / 2
    
    # Risk-neutral density - let it go negative if w is negative!
    density_basic = (1.0 / np.sqrt(2 * np.pi * w_safe * T)) * np.exp(-d2**2 / 2)
    
    # Apply sign correction based on total variance
    # If w < 0, this represents an inconsistent/arbitrageable situation
    density = np.where(w >= 0, density_basic, -density_basic)
    
    return density, w

def test_extreme_svi_parameters():
    """Test SVI parameters that should produce arbitrage opportunities."""
    k_vals = np.linspace(-10.0, 10.0, 1000)
    T = 0.5
    
    test_cases = [
        {"name": "Default (should be ok)", "a": 0.02, "b": 0.4, "rho": -0.2, "m": 0.0, "sigma": 0.4},
        {"name": "Extreme negative rho", "a": 0.02, "b": 0.8, "rho": -0.95, "m": 0.0, "sigma": 0.2},
        {"name": "Very high b parameter", "a": 0.01, "b": 2.0, "rho": -0.5, "m": 0.0, "sigma": 0.1},
        {"name": "Negative a parameter", "a": -0.1, "b": 0.5, "rho": -0.3, "m": 0.0, "sigma": 0.3},
        {"name": "Extreme combination", "a": -0.05, "b": 1.5, "rho": -0.9, "m": 2.0, "sigma": 0.1},
    ]
    
    fig, axes = plt.subplots(len(test_cases), 3, figsize=(15, 4*len(test_cases)))
    if len(test_cases) == 1:
        axes = axes.reshape(1, -1)
    
    for i, params in enumerate(test_cases):
        print(f"\n=== {params['name']} ===")
        a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']
        print(f"Parameters: a={a}, b={b}, rho={rho}, m={m}, sigma={sigma}")
        
        # Compute total variance
        w = compute_svi_total_variance(k_vals, a, b, rho, m, sigma)
        
        # Compute density (allowing negatives)
        density, _ = compute_risk_neutral_density_unclipped(k_vals, T, a, b, rho, m, sigma)
        
        # Compute volatility smile
        vol_smile = np.sqrt(np.maximum(w, 1e-8) / T)
        
        # Analysis
        print(f"Total variance range: [{np.min(w):.6f}, {np.max(w):.6f}]")
        print(f"Volatility range: [{np.min(vol_smile)*100:.1f}%, {np.max(vol_smile)*100:.1f}%]")
        
        negative_w = np.sum(w < 0)
        negative_density = np.sum(density < 0)
        min_density = np.min(density)
        
        if negative_w > 0:
            print(f"🚨 {negative_w} points with negative total variance!")
        if negative_density > 0:
            print(f"🚨 {negative_density} points with negative density!")
            print(f"Minimum density: {min_density:.6f}")
        
        # Integration
        positive_density = np.maximum(density, 0.0)
        total_prob = np.trapz(positive_density, k_vals)
        full_integral = np.trapz(density, k_vals)
        print(f"Positive probability integral: {total_prob:.6f}")
        print(f"Full integral (with negatives): {full_integral:.6f}")
        
        # Plot volatility
        axes[i, 0].plot(k_vals, vol_smile * 100, 'b-', linewidth=2)
        axes[i, 0].set_xlabel('log-moneyness')
        axes[i, 0].set_ylabel('Implied vol (%)')
        axes[i, 0].set_title(f'{params["name"]}\nVolatility Smile')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlim(-5, 5)
        
        # Plot total variance
        axes[i, 1].plot(k_vals, w, 'g-', linewidth=2)
        axes[i, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[i, 1].set_xlabel('log-moneyness')
        axes[i, 1].set_ylabel('Total variance')
        axes[i, 1].set_title('Total Variance')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim(-5, 5)
        
        # Plot density (with potential negatives!)
        axes[i, 2].plot(k_vals, density, 'r-', linewidth=2)
        axes[i, 2].axhline(y=0, color='k', linestyle='--', alpha=0.7)
        axes[i, 2].set_xlabel('log-moneyness')
        axes[i, 2].set_ylabel('Density')
        axes[i, 2].set_title('Risk-Neutral Density')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_xlim(-5, 5)
        
        # Highlight negative regions
        if negative_density > 0:
            negative_mask = density < 0
            axes[i, 2].fill_between(k_vals, 0, density, where=negative_mask, 
                                   color='red', alpha=0.3, label='ARBITRAGE!')
            axes[i, 2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_extreme_svi_parameters()
