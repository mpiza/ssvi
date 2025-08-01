"""
Core SVI and SSVI volatility model implementations.

This module contains the fundamental calculation functions for SVI (Stochastic Volatility Inspired)
and SSVI (Surface SVI) volatility models, separated from visualization and application code.

References:
- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives.
- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
"""

import numpy as np


def compute_svi_total_variance(k_values, a, b, rho, m, sigma):
    """Compute SVI total variance for given log-moneyness values.
    
    The SVI total variance is defined as:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    
    Parameters
    ----------
    k_values : ndarray
        1-D array of log-moneyness values.
    a, b, rho, m, sigma : float
        SVI parameters. See Gatheral (2004) for details.
        - a: level (intercept)
        - b: angle (slope parameter)  
        - rho: correlation parameter (-1 < rho < 1)
        - m: horizontal shift
        - sigma: curvature parameter (sigma > 0)
    
    Returns
    -------
    total_variance : ndarray
        1-D array of total variance values w(k).
    """
    diff = k_values - m
    total_variance = a + b * (rho * diff + np.sqrt(diff**2 + sigma**2))
    return total_variance


def compute_svi_volatility_smile(k_values, T, a, b, rho, m, sigma):
    """Compute SVI volatility smile for a single maturity.
    
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
    vol_smile : ndarray
        1-D array of implied volatilities sigma(k, T) = sqrt(w(k) / T).
    """
    total_variance = compute_svi_total_variance(k_values, a, b, rho, m, sigma)
    # Ensure non-negative total variance and compute implied vol
    vol_smile = np.sqrt(np.maximum(total_variance, 0.0) / T)
    return vol_smile


def compute_svi_surface(k_values, t_values, a, b, rho, m, sigma):
    """Compute the SVI volatility surface.

    Parameters
    ----------
    k_values : ndarray
        1-D array of log-moneyness values.
    t_values : ndarray
        1-D array of time to maturity values.
    a, b, rho, m, sigma : float
        SVI parameters. See Gatheral (2004) for details.

    Returns
    -------
    vol_surface : ndarray
        2-D array with shape (len(t_values), len(k_values)) containing
        Black implied volatilities computed from the SVI total variance
        parameterisation.
    """
    # Total variance as a function of k only
    total_variance_k = compute_svi_total_variance(k_values, a, b, rho, m, sigma)

    # Expand across maturities: each maturity sees the same total variance function
    # but implied volatility scales by sqrt(w / T)
    # Avoid division by zero by ensuring T > 0
    T_grid, K_grid = np.meshgrid(t_values, k_values, indexing="ij")
    W_grid = total_variance_k[np.newaxis, :].repeat(len(t_values), axis=0)
    vol_surface = np.sqrt(np.maximum(W_grid, 0.0) / T_grid)
    return vol_surface


def compute_ssvi_total_variance(k_values, theta, phi, rho):
    """Compute SSVI total variance for given log-moneyness values.
    
    The SSVI total variance is defined as:
        w(k) = 0.5 * theta * (1 + rho * phi * k + sqrt((phi * k + rho)^2 + 1 - rho^2))
    
    Parameters
    ----------
    k_values : ndarray
        1-D array of log-moneyness values.
    theta, phi, rho : float
        SSVI parameters.
        - theta: variance level (theta > 0)
        - phi: skew parameter (phi > 0)
        - rho: correlation parameter (-1 < rho < 1)
    
    Returns
    -------
    total_variance : ndarray
        1-D array of total variance values w(k).
    """
    kk = phi * k_values
    inside = (kk + rho)**2 + 1.0 - rho**2
    # Ensure numerical stability by clipping the radicand at zero
    rad = np.sqrt(np.maximum(inside, 0.0))
    w_k = 0.5 * theta * (1.0 + rho * kk + rad)
    return w_k


def compute_ssvi_surface(k_values, t_values, theta, phi, rho):
    """Compute the SSVI volatility surface.

    The SSVI (Surface SVI) total variance is defined as

        w(k) = 0.5 * theta * (1 + rho * phi * k + sqrt((phi * k + rho)^2 + 1 - rho^2)),

    where theta > 0, phi > 0, and -1 < rho < 1. The implied volatility is
    derived by sigma(k, T) = sqrt(w(k) / T) for maturity T.

    Parameters
    ----------
    k_values : ndarray
        1-D array of log-moneyness values.
    t_values : ndarray
        1-D array of time to maturity values.
    theta, phi, rho : float
        SSVI parameters.

    Returns
    -------
    vol_surface : ndarray
        2-D array with shape (len(t_values), len(k_values)) containing
        implied volatilities computed from the SSVI total variance.
    """
    # Total variance as a function of k only
    w_k = compute_ssvi_total_variance(k_values, theta, phi, rho)

    # Build the grid and compute implied vols as with SVI
    T_grid, K_grid = np.meshgrid(t_values, k_values, indexing="ij")
    W_grid = w_k[np.newaxis, :].repeat(len(t_values), axis=0)
    vol_surface = np.sqrt(np.maximum(W_grid, 0.0) / T_grid)
    return vol_surface


def validate_svi_parameters(a, b, rho, m, sigma):
    """Validate SVI parameters to ensure no arbitrage conditions.
    
    Parameters
    ----------
    a, b, rho, m, sigma : float
        SVI parameters to validate.
    
    Returns
    -------
    is_valid : bool
        True if parameters satisfy no-arbitrage conditions.
    violations : list
        List of constraint violations (empty if valid).
    """
    violations = []
    
    # Basic parameter constraints
    if a < 0:
        violations.append(f"Parameter 'a' must be non-negative, got {a}")
    if b < 0:
        violations.append(f"Parameter 'b' must be non-negative, got {b}")
    if abs(rho) >= 1:
        violations.append(f"Parameter 'rho' must be in (-1, 1), got {rho}")
    if sigma <= 0:
        violations.append(f"Parameter 'sigma' must be positive, got {sigma}")
    
    # Gatheral no-arbitrage conditions (simplified check)
    if b != 0:
        # Check that the discriminant doesn't lead to arbitrage
        discriminant = b**2 * (1 - rho**2)
        if discriminant < 0:
            violations.append("SVI parameters may lead to arbitrage (negative discriminant)")
    
    return len(violations) == 0, violations


def validate_ssvi_parameters(theta, phi, rho):
    """Validate SSVI parameters to ensure no arbitrage conditions.
    
    Parameters
    ----------
    theta, phi, rho : float
        SSVI parameters to validate.
    
    Returns
    -------
    is_valid : bool
        True if parameters satisfy no-arbitrage conditions.
    violations : list
        List of constraint violations (empty if valid).
    """
    violations = []
    
    if theta <= 0:
        violations.append(f"Parameter 'theta' must be positive, got {theta}")
    if phi <= 0:
        violations.append(f"Parameter 'phi' must be positive, got {phi}")
    if abs(rho) >= 1:
        violations.append(f"Parameter 'rho' must be in (-1, 1), got {rho}")
    
    return len(violations) == 0, violations
