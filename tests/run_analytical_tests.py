#!/usr/bin/env python3
"""
Run all analytical derivative tests for the parametric SSVI implementation.

This script runs the complete test suite for analytical derivatives and provides
a summary of all test results.
"""

import subprocess
import sys
import os

def run_test(test_script):
    """Run a test script and return the result."""
    try:
        print(f"\n{'='*60}")
        print(f"Running: {test_script}")
        print('='*60)
        
        result = subprocess.run([sys.executable, test_script], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(f"✅ {test_script} PASSED")
            return True
        else:
            print(f"❌ {test_script} FAILED")
            return False
            
    except Exception as e:
        print(f"❌ {test_script} ERROR: {e}")
        return False

def main():
    """Run all analytical derivative tests."""
    print("Parametric SSVI Analytical Derivatives Test Suite")
    print("=" * 60)
    
    # List of test scripts to run
    test_scripts = [
        "test_phi_derivative_accuracy.py",
        "test_analytical_derivatives.py",
    ]
    
    results = []
    
    for test_script in test_scripts:
        if os.path.exists(test_script):
            success = run_test(test_script)
            results.append((test_script, success))
        else:
            print(f"⚠️  {test_script} not found, skipping...")
            results.append((test_script, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_script, success in results:
        status = "PASS" if success else "FAIL"
        icon = "✅" if success else "❌"
        print(f"{icon} {test_script:<35} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All analytical derivative tests PASSED!")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
