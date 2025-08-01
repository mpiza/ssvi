#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Quick example showing how to plot implied volatility vs local volatility
"""

import numpy as np
import matplotlib.pyplot as plt
from src.svi_models import compute_svi_volatility_smile
from src.local_volatility import compute_svi_local_volatility

# Parameters
k_values = np.linspace(-2.0, 2.0, 100)
T = 1.0
a, b, rho, m, sigma = 0.04, 0.3, -0.3, 0.0, 0.4
r = 0.02

print("Computing implied and local volatilities...")

# Compute implied volatility
iv = compute_svi_volatility_smile(k_values, T, a, b, rho, m, sigma)

# Compute local volatility
lv, is_valid, diagnostics = compute_svi_local_volatility(
    k_values, T, a, b, rho, m, sigma, r=r
)

# Apply validity mask
lv_masked = np.where(is_valid, lv, np.nan)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top: Both volatilities
ax1.plot(k_values, iv, 'b-', linewidth=2, label='Implied Volatility', alpha=0.8)
ax1.plot(k_values, lv_masked, 'r-', linewidth=2, label='Local Volatility', alpha=0.8)
ax1.set_xlabel('Log-Moneyness k')
ax1.set_ylabel('Volatility')
ax1.set_title('Implied vs Local Volatility Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom: Ratio
valid_iv = iv[is_valid]
valid_lv = lv[is_valid]
valid_k = k_values[is_valid]

if len(valid_iv) > 0:
    ratio = valid_lv / valid_iv
    ax2.plot(valid_k, ratio, 'g-', linewidth=2, label='Local/Implied Ratio')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='LV = IV')
    ax2.set_xlabel('Log-Moneyness k')
    ax2.set_ylabel('LV/IV Ratio')
    ax2.set_title('Local/Implied Volatility Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\n=== Summary ===")
print(f"Valid local volatility points: {np.sum(is_valid)}/{len(k_values)}")
print(f"Implied volatility range: {np.min(iv):.4f} - {np.max(iv):.4f}")
if np.sum(is_valid) > 0:
    print(f"Local volatility range: {np.min(valid_lv):.4f} - {np.max(valid_lv):.4f}")
    print(f"LV/IV ratio range: {np.min(ratio):.4f} - {np.max(ratio):.4f}")
