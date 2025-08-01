"""
Risk-neutral density analysis for SVI volatility models.

This module contains functions for computing and validating risk-neutral probability
densities derived from SVI volatility surfaces using the Breeden-Litzenberger formula.
"""

import numpy as np
from .svi_models import compute_svi_total_variance


def compute_svi_risk_neutral_density(k_values, T, a, b, rho, m, sigma):
    """Compute risk-neutral density from SVI total variance using the Breeden-Litzenberger formula.
    
    This implements the complete derivation with the shape function g(y) that accounts for
    the curvature and derivatives of the SVI total variance function.
    
    Following the derivation:
    w(y) = a + b[ρ(y-m) + √((y-m)² + σ²)]
    w'(y) = b[ρ + (y-m)/√((y-m)² + σ²)]
    w''(y) = b σ² / ((y-m)² + σ²)^(3/2)
    
    g(y) = [1 - y w'(y)/(2w(y))]² - (w'(y))²/4 [1/w(y) + 1/4] + w''(y)/2
    
    p(y) = g(y)/√(2πw(y)) * exp(-d₋²/2)
    where d₋ = -y/√w(y) - √w(y)/2
    
    For log-moneyness density: ρ(k) = p(k)
    
    Parameters
    ----------
    k_values : ndarray
        1-D array of log-moneyness values.
    T : float
        Time to maturity in years (T > 0).
    a, b, rho, m, sigma : float
        SVI parameters.
    
    Returns
    -------
    density : ndarray
        1-D array of risk-neutral density values in log-moneyness space.
    """
    # SVI total variance and its derivatives
    y = k_values  # log-moneyness (using k instead of y for consistency with code)
    diff = y - m
    sqrt_term = np.sqrt(diff**2 + sigma**2)
    
    # Total variance w(y)
    w = compute_svi_total_variance(k_values, a, b, rho, m, sigma)
    
    # Ensure positive total variance for numerical stability
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
    
    # Risk-neutral density in log-moneyness space p(y)
    # This is the CORRECT formula with the shape function
    density = (g / np.sqrt(2 * np.pi * w)) * np.exp(-d_minus**2 / 2)
    
    return density


def verify_density_properties(k_values, density, verbose=True):
    """Verify that the computed density has proper mathematical properties.
    
    Parameters
    ----------
    k_values : ndarray
        1-D array of log-moneyness values.
    density : ndarray
        1-D array of density values corresponding to k_values.
    verbose : bool, optional
        If True, print detailed verification results. Default is True.
    
    Returns
    -------
    verification_results : dict
        Dictionary containing verification results:
        - 'has_negative': bool, whether density has negative values
        - 'total_probability': float, integral of positive part
        - 'full_integral': float, integral including negative parts  
        - 'min_density': float, minimum density value
        - 'has_infinite': bool, whether density contains infinite values
        - 'has_nan': bool, whether density contains NaN values
        - 'expected_k': float, expected log-moneyness (if meaningful)
        - 'is_valid': bool, overall validity assessment
    """
    results = {}
    
    # Test 1: Check for negative densities (ARBITRAGE INDICATOR!)
    negative_mask = density < 0
    results['has_negative'] = np.any(negative_mask)
    results['min_density'] = np.min(density)
    
    if results['has_negative'] and verbose:
        num_negative = np.sum(negative_mask)
        negative_k_range = [np.min(k_values[negative_mask]), np.max(k_values[negative_mask])]
        print(f"🚨 ARBITRAGE DETECTED! 🚨")
        print(f"Found {num_negative} points with NEGATIVE density!")
        print(f"Minimum density: {results['min_density']:.6f}")
        print(f"Negative density occurs in k-range: [{negative_k_range[0]:.2f}, {negative_k_range[1]:.2f}]")
    elif verbose:
        print(f"✓ All density values are non-negative")
    
    # Test 2: Probability conservation (should integrate to 1)
    # Only integrate positive part for meaningful probability
    positive_density = np.maximum(density, 0.0)
    results['total_probability'] = np.trapz(positive_density, k_values)
    
    # Test 3: Full integral (including negative parts)
    results['full_integral'] = np.trapz(density, k_values)
    
    # Test 4: Finite values
    results['has_infinite'] = np.any(np.isinf(density))
    results['has_nan'] = np.any(np.isnan(density))
    
    # Test 5: Expected value (first moment, only positive part)
    if results['total_probability'] > 0:
        results['expected_k'] = np.trapz(k_values * positive_density, k_values) / results['total_probability']
    else:
        results['expected_k'] = np.nan
    
    # Overall validity
    results['is_valid'] = (
        not results['has_negative'] and 
        not results['has_infinite'] and 
        not results['has_nan'] and
        0.8 <= results['total_probability'] <= 1.2  # Allow some numerical error
    )
    
    if verbose:
        print(f"Total probability (positive part): {results['total_probability']:.6f}")
        print(f"Full integral (including negatives): {results['full_integral']:.6f}")
        print(f"Contains infinite values: {results['has_infinite']}")
        print(f"Contains NaN values: {results['has_nan']}")
        if not np.isnan(results['expected_k']):
            print(f"Expected log-moneyness (positive part): {results['expected_k']:.6f}")
    
    return results


def analyze_density_arbitrage(k_values, density, T, a, b, rho, m, sigma):
    """Analyze density for arbitrage opportunities and provide diagnostic information.
    
    Parameters
    ----------
    k_values : ndarray
        1-D array of log-moneyness values.
    density : ndarray
        1-D array of density values.
    T : float
        Time to maturity.
    a, b, rho, m, sigma : float
        SVI parameters used to generate the density.
    
    Returns
    -------
    analysis : dict
        Dictionary containing detailed arbitrage analysis.
    """
    analysis = {}
    
    # Basic density verification
    verification = verify_density_properties(k_values, density, verbose=False)
    analysis['verification'] = verification
    
    # Parameter analysis
    analysis['parameters'] = {
        'T': T, 'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma
    }
    
    # Identify problematic regions
    if verification['has_negative']:
        negative_mask = density < 0
        negative_indices = np.where(negative_mask)[0]
        analysis['negative_regions'] = {
            'indices': negative_indices,
            'k_values': k_values[negative_mask],
            'density_values': density[negative_mask],
            'severity': np.sum(np.abs(density[negative_mask]))
        }
    
    # Check slope constraints (simplified butterfly arbitrage check)
    if len(k_values) > 2:
        # Compute second derivative of call prices (related to density)
        dk = k_values[1] - k_values[0]  # Assume uniform grid
        if dk > 0:
            # Approximate second derivative
            d2_density = np.gradient(np.gradient(density, dk), dk)
            negative_curvature_mask = d2_density < -1e-6  # Small tolerance for numerical errors
            
            if np.any(negative_curvature_mask):
                analysis['butterfly_violations'] = {
                    'indices': np.where(negative_curvature_mask)[0],
                    'k_values': k_values[negative_curvature_mask],
                    'curvature': d2_density[negative_curvature_mask]
                }
    
    # Overall arbitrage assessment
    has_arbitrage = (
        verification['has_negative'] or 
        verification['has_infinite'] or 
        verification['has_nan'] or
        'butterfly_violations' in analysis
    )
    
    analysis['has_arbitrage'] = has_arbitrage
    analysis['arbitrage_score'] = _compute_arbitrage_score(verification, analysis)
    
    return analysis


def _compute_arbitrage_score(verification, analysis):
    """Compute a numerical score indicating the severity of arbitrage violations.
    
    Returns a score between 0 (no arbitrage) and 1 (severe arbitrage).
    """
    score = 0.0
    
    # Negative density penalty
    if verification['has_negative']:
        score += 0.5 * min(1.0, abs(verification['min_density']) / 0.1)
    
    # Probability conservation penalty
    prob_error = abs(verification['total_probability'] - 1.0)
    score += 0.2 * min(1.0, prob_error / 0.2)
    
    # Infinite/NaN penalty
    if verification['has_infinite'] or verification['has_nan']:
        score += 0.3
    
    # Butterfly arbitrage penalty
    if 'butterfly_violations' in analysis:
        score += 0.2
    
    return min(1.0, score)


def compute_moment_from_density(k_values, density, moment=1):
    """Compute moments of the risk-neutral distribution.
    
    Parameters
    ----------
    k_values : ndarray
        1-D array of log-moneyness values.
    density : ndarray
        1-D array of density values (should be non-negative).
    moment : int, optional
        Which moment to compute (1=mean, 2=second moment, etc.). Default is 1.
    
    Returns
    -------
    moment_value : float
        The computed moment, or NaN if density is invalid.
    """
    # Use only non-negative part of density for meaningful probability calculation
    positive_density = np.maximum(density, 0.0)
    total_prob = np.trapz(positive_density, k_values)
    
    if total_prob <= 0:
        return np.nan
    
    # Normalize density
    normalized_density = positive_density / total_prob
    
    # Compute moment
    if moment == 1:
        return np.trapz(k_values * normalized_density, k_values)
    else:
        return np.trapz((k_values ** moment) * normalized_density, k_values)
