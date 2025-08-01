#!/usr/bin/env python3
"""
Test local volatility computation for the constant implied volatility case.

In the case of constant implied volatility (flat smile), the local volatility
should exactly equal the implied volatility. This is a fundamental test case
that validates the Dupire formula implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from svi_models import compute_svi_volatility_smile
from local_volatility import compute_svi_local_volatility

def create_flat_svi_parameters(target_vol=0.20, T=1.0):
    """
    Create SVI parameters that produce a flat (constant) volatility smile.
    
    For a flat smile, we want σ_IV(k) = target_vol for all k.
    Since σ_IV = √(w(k)/T), we need w(k) = target_vol² * T = constant.
    This happens when the SVI function reduces to just the constant term 'a'.
    
    Setting b = 0 makes the SVI formula: w(k) = a
    So we set a = target_vol² * T and b = 0.
    """
    return {
        'a': target_vol**2 * T,  # Total variance = σ² * T
        'b': 0.0,                # No slope -> flat smile
        'rho': 0.0,              # Doesn't matter when b=0
        'm': 0.0,                # Doesn't matter when b=0  
        'sigma': 0.1,            # Needs to be > 0 for validity, but doesn't affect result when b=0
    }

def test_constant_volatility_case():
    """Test that local volatility equals implied volatility for constant smile."""
    
    print("=== Testing Constant Implied Volatility Case ===")
    print()
    
    # Test parameters
    target_vol = 0.25  # 25% volatility
    T = 1.0           # 1 year maturity
    r = 0.0           # Zero interest rate for simplest case
    
    # Create log-moneyness grid
    k_vals = np.linspace(-2.0, 2.0, 101)
    
    # Get SVI parameters for flat smile
    svi_params = create_flat_svi_parameters(target_vol, T)
    
    print(f"Target constant volatility: {target_vol:.4f} ({target_vol*100:.2f}%)")
    print(f"SVI parameters for flat smile:")
    for param, value in svi_params.items():
        print(f"  {param}: {value:.6f}")
    print()
    
    # Compute implied volatility smile
    implied_vol = compute_svi_volatility_smile(k_vals, T=T, **svi_params)
    
    # Compute local volatility
    local_vol, is_valid, diagnostics = compute_svi_local_volatility(
        k_vals, T=T, r=r, **svi_params
    )
    
    # Analysis
    print("=== Results ===")
    print(f"Implied volatility (should be constant {target_vol:.4f}):")
    print(f"  Min: {np.min(implied_vol):.6f}")
    print(f"  Max: {np.max(implied_vol):.6f}")
    print(f"  Range: {np.max(implied_vol) - np.min(implied_vol):.8f}")
    print()
    
    valid_count = np.sum(is_valid)
    print(f"Local volatility validity: {valid_count}/{len(k_vals)} points valid")
    
    if valid_count > 0:
        valid_local_vol = local_vol[is_valid]
        valid_implied_vol = implied_vol[is_valid]
        
        print(f"Local volatility (valid points only):")
        print(f"  Min: {np.min(valid_local_vol):.6f}")
        print(f"  Max: {np.max(valid_local_vol):.6f}")
        print(f"  Range: {np.max(valid_local_vol) - np.min(valid_local_vol):.8f}")
        print()
        
        # Compute differences
        diff = valid_local_vol - valid_implied_vol
        print(f"Difference (local - implied):")
        print(f"  Mean: {np.mean(diff):.8f}")
        print(f"  Std: {np.std(diff):.8f}")
        print(f"  Max abs: {np.max(np.abs(diff)):.8f}")
        print()
        
        # Check if they match within tolerance
        tolerance = 1e-6
        matches = np.allclose(valid_local_vol, valid_implied_vol, atol=tolerance, rtol=tolerance)
        print(f"Local vol matches implied vol (tolerance {tolerance:.0e}): {matches}")
        
        if not matches:
            print("❌ TEST FAILED: Local volatility does not match implied volatility!")
            print("This indicates an issue with the Dupire formula implementation.")
        else:
            print("✅ TEST PASSED: Local volatility matches implied volatility.")
    else:
        print("❌ TEST FAILED: No valid local volatility points computed!")
        matches = False
    
    print()
    print("=== Diagnostics ===")
    for key, value in diagnostics.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: array with {len(value)} elements")
            if len(value) > 0:
                print(f"  Min: {np.min(value):.6f}, Max: {np.max(value):.6f}")
        else:
            print(f"{key}: {value}")
    
    # Skip plotting for now to avoid hanging
    print("Skipping interactive plots to avoid hanging...")
    
    return matches

if __name__ == "__main__":
    # Run the constant volatility test
    success = test_constant_volatility_case()
    
    print()
    print("=== Summary ===")
    if success:
        print("✅ Constant volatility test PASSED")
        print("The Dupire formula implementation correctly handles the flat smile case.")
    else:
        print("❌ Constant volatility test FAILED")
        print("There is likely an issue with the Dupire formula implementation.")
        print("Specifically, check:")
        print("1. Time derivative ∂w/∂T handling")
        print("2. Numerical computation of spatial derivatives")
        print("3. Denominator calculation in Dupire formula")
