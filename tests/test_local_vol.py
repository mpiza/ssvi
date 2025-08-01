#!/usr/bin/env python3
"""
Simple test of local volatility computation
"""

import numpy as np
from src.local_volatility import (
    compute_svi_local_volatility,
    compute_ssvi_local_volatility,
    print_local_volatility_analysis,
    analyze_local_volatility_properties
)

def test_svi_local_volatility():
    """Test SVI local volatility computation."""
    print("Testing SVI Local Volatility...")
    
    # Simple test case
    k_values = np.linspace(-1.0, 1.0, 100)
    T = 0.5
    a, b, rho, m, sigma = 0.04, 0.4, -0.3, 0.0, 0.3
    r = 0.02
    
    # Compute local volatility
    local_vol, is_valid, diagnostics = compute_svi_local_volatility(
        k_values, T, a, b, rho, m, sigma, r=r
    )
    
    # Basic checks
    print(f"Total points: {len(k_values)}")
    print(f"Valid points: {np.sum(is_valid)} ({100*np.sum(is_valid)/len(k_values):.1f}%)")
    
    if np.any(is_valid):
        valid_local_vol = local_vol[is_valid]
        print(f"Local vol range: {np.min(valid_local_vol):.4f} - {np.max(valid_local_vol):.4f}")
        
        # Find ATM value
        atm_idx = np.argmin(np.abs(k_values))
        if is_valid[atm_idx]:
            print(f"ATM local volatility: {local_vol[atm_idx]:.4f}")
        
        # Compare to implied volatility at ATM
        implied_vol_atm = np.sqrt(diagnostics['total_variance'][atm_idx] / T)
        print(f"ATM implied volatility: {implied_vol_atm:.4f}")
        
        if is_valid[atm_idx]:
            ratio = local_vol[atm_idx] / implied_vol_atm
            print(f"Local/Implied ratio at ATM: {ratio:.4f}")
    
    # Analysis
    analysis = analyze_local_volatility_properties(k_values, local_vol, is_valid, "SVI")
    print_local_volatility_analysis(analysis)
    
    return True

def test_ssvi_local_volatility():
    """Test SSVI local volatility computation."""
    print("Testing SSVI Local Volatility...")
    
    # Simple test case
    k_values = np.linspace(-1.0, 1.0, 100)
    T = 0.25
    theta, phi, rho = 0.2, 1.0, -0.4
    r = 0.03
    
    # Compute local volatility
    local_vol, is_valid, diagnostics = compute_ssvi_local_volatility(
        k_values, T, theta, phi, rho, r=r
    )
    
    # Basic checks
    print(f"Total points: {len(k_values)}")
    print(f"Valid points: {np.sum(is_valid)} ({100*np.sum(is_valid)/len(k_values):.1f}%)")
    
    if np.any(is_valid):
        valid_local_vol = local_vol[is_valid]
        print(f"Local vol range: {np.min(valid_local_vol):.4f} - {np.max(valid_local_vol):.4f}")
        
        # Find ATM value
        atm_idx = np.argmin(np.abs(k_values))
        if is_valid[atm_idx]:
            print(f"ATM local volatility: {local_vol[atm_idx]:.4f}")
        
        # Compare to implied volatility at ATM
        implied_vol_atm = np.sqrt(diagnostics['total_variance'][atm_idx] / T)
        print(f"ATM implied volatility: {implied_vol_atm:.4f}")
    
    # Analysis
    analysis = analyze_local_volatility_properties(k_values, local_vol, is_valid, "SSVI")
    print_local_volatility_analysis(analysis)
    
    return True

def main():
    """Run basic tests."""
    print("Local Volatility - Basic Tests")
    print("=" * 40)
    
    try:
        test_svi_local_volatility()
        print()
        test_ssvi_local_volatility()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
