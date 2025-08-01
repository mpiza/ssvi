#!/usr/bin/env python3
"""
Parametric SSVI Model Implementation

This module implements the extended SSVI model with parametric time dependence:
- Time-dependent variance level θ(T) with mean reversion
- Rational function φ(θ) for time-dependent skew
- Complete parameter set for flexible surface modeling

Mathematical Framework:
w(μ, T) = (θT/2) * [1 + ρφ(θT)μ + √((φ(θT)μ + ρ)² + (1 - ρ²))]

Where:
- θT = θ∞T + (θ0 - θ∞) * (1 - e^(-κT))/κ
- φ(θ) = (p0 + p1*θ + p2*θ²)/(q0 + q1*θ + q2*θ²)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import warnings


def compute_theta_T(T: np.ndarray, theta_inf: float, theta_0: float, kappa: float) -> np.ndarray:
    """
    Compute time-dependent variance level θ(T).
    
    θT = θ∞T + (θ0 - θ∞) * (1 - e^(-κT))/κ
    
    Parameters:
    -----------
    T : np.ndarray
        Time to maturity values
    theta_inf : float
        Long-term variance level (θ∞)
    theta_0 : float
        Initial variance level (θ0)
    kappa : float
        Mean reversion speed (κ > 0)
        
    Returns:
    --------
    np.ndarray
        Time-dependent variance level θ(T)
    """
    T = np.asarray(T)
    
    # Handle the case κ → 0 (no mean reversion, linear growth)
    if abs(kappa) < 1e-10:
        return theta_0 * T
    
    # Standard mean reversion formula
    exp_term = np.exp(-kappa * T)
    theta_T = theta_inf * T + (theta_0 - theta_inf) * (1 - exp_term) / kappa
    
    return theta_T


def compute_phi_rational(theta: np.ndarray, p_coeffs: List[float], q_coeffs: List[float]) -> np.ndarray:
    """
    Compute rational function φ(θ) = P(θ)/Q(θ).
    
    φ(θ) = (p0 + p1*θ + p2*θ²)/(q0 + q1*θ + q2*θ²)
    
    Parameters:
    -----------
    theta : np.ndarray
        Variance level values
    p_coeffs : List[float]
        Numerator coefficients [p0, p1, p2]
    q_coeffs : List[float]
        Denominator coefficients [q0, q1, q2]
        
    Returns:
    --------
    np.ndarray
        Rational function values φ(θ)
    """
    theta = np.asarray(theta)
    
    # Evaluate polynomials
    p0, p1, p2 = p_coeffs
    q0, q1, q2 = q_coeffs
    
    numerator = p0 + p1 * theta + p2 * theta**2
    denominator = q0 + q1 * theta + q2 * theta**2
    
    # Check for near-zero denominators
    small_denom = np.abs(denominator) < 1e-12
    if np.any(small_denom):
        warnings.warn("Small denominator values detected in φ(θ) computation")
        denominator = np.where(small_denom, 1e-12, denominator)
    
    phi = numerator / denominator
    
    return phi


def compute_phi_rational_derivative(theta: np.ndarray, p_coeffs: List[float], q_coeffs: List[float]) -> np.ndarray:
    """
    Compute analytical derivative of rational function φ(θ) with respect to θ.
    
    ∂φ/∂θ = [P'(θ)Q(θ) - P(θ)Q'(θ)] / [Q(θ)]²
    
    Where:
    - P(θ) = p₀ + p₁θ + p₂θ²
    - Q(θ) = q₀ + q₁θ + q₂θ²  
    - P'(θ) = p₁ + 2p₂θ
    - Q'(θ) = q₁ + 2q₂θ
    
    Parameters:
    -----------
    theta : np.ndarray
        Variance level values
    p_coeffs : List[float]
        Numerator coefficients [p0, p1, p2]
    q_coeffs : List[float]
        Denominator coefficients [q0, q1, q2]
        
    Returns:
    --------
    np.ndarray
        Derivative values ∂φ/∂θ
    """
    theta = np.asarray(theta)
    
    # Extract coefficients
    p0, p1, p2 = p_coeffs
    q0, q1, q2 = q_coeffs
    
    # Compute polynomials and their derivatives
    P_theta = p0 + p1 * theta + p2 * theta**2
    Q_theta = q0 + q1 * theta + q2 * theta**2
    P_prime = p1 + 2 * p2 * theta
    Q_prime = q1 + 2 * q2 * theta
    
    # Quotient rule: [P'(θ)Q(θ) - P(θ)Q'(θ)] / [Q(θ)]²
    numerator = P_prime * Q_theta - P_theta * Q_prime
    denominator = Q_theta**2
    
    # Check for near-zero denominators
    small_denom = np.abs(denominator) < 1e-12
    if np.any(small_denom):
        warnings.warn("Small denominator values detected in ∂φ/∂θ computation")
        denominator = np.where(small_denom, 1e-12, denominator)
    
    dphi_dtheta = numerator / denominator
    
    return dphi_dtheta


def compute_parametric_ssvi_total_variance(
    mu_values: np.ndarray, 
    T: float, 
    rho: float,
    theta_inf: float, 
    theta_0: float, 
    kappa: float,
    p_coeffs: List[float], 
    q_coeffs: List[float]
) -> np.ndarray:
    """
    Compute parametric SSVI total variance.
    
    w(μ, T) = (θT/2) * [1 + ρφ(θT)μ + √((φ(θT)μ + ρ)² + (1 - ρ²))]
    
    Parameters:
    -----------
    mu_values : np.ndarray
        Log-moneyness values (μ = k in standard notation)
    T : float
        Time to maturity
    rho : float
        Correlation parameter (-1 < ρ < 1)
    theta_inf : float
        Long-term variance level (θ∞)
    theta_0 : float
        Initial variance level (θ0)
    kappa : float
        Mean reversion speed (κ > 0)
    p_coeffs : List[float]
        Numerator coefficients [p0, p1, p2] for φ(θ)
    q_coeffs : List[float]
        Denominator coefficients [q0, q1, q2] for φ(θ)
        
    Returns:
    --------
    np.ndarray
        Total variance w(μ, T)
    """
    mu_values = np.asarray(mu_values)
    
    # Compute time-dependent variance level
    theta_T = compute_theta_T(np.array([T]), theta_inf, theta_0, kappa)[0]
    
    # Compute rational function φ(θT)
    phi_T = compute_phi_rational(np.array([theta_T]), p_coeffs, q_coeffs)[0]
    
    # SSVI formula
    term1 = rho * phi_T * mu_values
    term2 = np.sqrt((phi_T * mu_values + rho)**2 + (1 - rho**2))
    
    w = (theta_T / 2) * (1 + term1 + term2)
    
    return w


def compute_parametric_ssvi_surface(
    mu_values: np.ndarray,
    T_values: np.ndarray,
    rho: float,
    theta_inf: float,
    theta_0: float,
    kappa: float,
    p_coeffs: List[float],
    q_coeffs: List[float]
) -> np.ndarray:
    """
    Compute parametric SSVI total variance surface.
    
    Parameters:
    -----------
    mu_values : np.ndarray
        Log-moneyness values
    T_values : np.ndarray
        Time to maturity values
    rho : float
        Correlation parameter (-1 < ρ < 1)
    theta_inf : float
        Long-term variance level (θ∞)
    theta_0 : float
        Initial variance level (θ0)
    kappa : float
        Mean reversion speed (κ > 0)
    p_coeffs : List[float]
        Numerator coefficients [p0, p1, p2] for φ(θ)
    q_coeffs : List[float]
        Denominator coefficients [q0, q1, q2] for φ(θ)
        
    Returns:
    --------
    np.ndarray
        Total variance surface w(μ, T) with shape (len(T_values), len(mu_values))
    """
    mu_values = np.asarray(mu_values)
    T_values = np.asarray(T_values)
    
    # Initialize surface
    surface = np.zeros((len(T_values), len(mu_values)))
    
    # Compute for each maturity
    for i, T in enumerate(T_values):
        surface[i, :] = compute_parametric_ssvi_total_variance(
            mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
        )
    
    return surface


def compute_parametric_ssvi_volatility_smile(
    mu_values: np.ndarray,
    T: float,
    rho: float,
    theta_inf: float,
    theta_0: float,
    kappa: float,
    p_coeffs: List[float],
    q_coeffs: List[float]
) -> np.ndarray:
    """
    Compute parametric SSVI implied volatility smile.
    
    Parameters:
    -----------
    mu_values : np.ndarray
        Log-moneyness values
    T : float
        Time to maturity
    rho : float
        Correlation parameter (-1 < ρ < 1)
    theta_inf : float
        Long-term variance level (θ∞)
    theta_0 : float
        Initial variance level (θ0)
    kappa : float
        Mean reversion speed (κ > 0)
    p_coeffs : List[float]
        Numerator coefficients [p0, p1, p2] for φ(θ)
    q_coeffs : List[float]
        Denominator coefficients [q0, q1, q2] for φ(θ)
        
    Returns:
    --------
    np.ndarray
        Implied volatility σ(μ, T)
    """
    # Compute total variance
    w = compute_parametric_ssvi_total_variance(
        mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    
    # Convert to volatility: σ = √(w/T)
    volatility = np.sqrt(np.maximum(w / T, 1e-12))
    
    return volatility


def compute_parametric_ssvi_derivatives(
    mu_values: np.ndarray,
    T: float,
    rho: float,
    theta_inf: float,
    theta_0: float,
    kappa: float,
    p_coeffs: List[float],
    q_coeffs: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute analytical derivatives of parametric SSVI total variance w.r.t. μ.
    
    Returns ∂w/∂μ and ∂²w/∂μ² using analytical formulas.
    
    Mathematical derivation:
    w(μ, T) = (θT/2) * [1 + ρφ(θT)μ + √((φ(θT)μ + ρ)² + (1 - ρ²))]
    
    Let z = φ(θT)μ + ρ, then:
    ∂w/∂μ = (θT/2) * [ρφ(θT) + φ(θT) * z/√(z² + (1 - ρ²))]
    ∂²w/∂μ² = (θT/2) * φ(θT)² * (1 - ρ²) / (z² + (1 - ρ²))^(3/2)
    
    Parameters:
    -----------
    mu_values : np.ndarray
        Log-moneyness values
    T : float
        Time to maturity
    rho : float
        Correlation parameter
    theta_inf : float
        Long-term variance level
    theta_0 : float
        Initial variance level
    kappa : float
        Mean reversion speed
    p_coeffs : List[float]
        Numerator coefficients for φ(θ)
    q_coeffs : List[float]
        Denominator coefficients for φ(θ)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (w, ∂w/∂μ, ∂²w/∂μ²)
    """
    mu_values = np.asarray(mu_values)
    
    # Compute time-dependent variance level
    theta_T = compute_theta_T(np.array([T]), theta_inf, theta_0, kappa)[0]
    
    # Compute rational function φ(θT)
    phi_T = compute_phi_rational(np.array([theta_T]), p_coeffs, q_coeffs)[0]
    
    # Intermediate variables
    z = phi_T * mu_values + rho  # φ(θT)μ + ρ
    sqrt_term = np.sqrt(z**2 + (1 - rho**2))
    
    # Total variance: w(μ, T)
    w = (theta_T / 2) * (1 + rho * phi_T * mu_values + sqrt_term)
    
    # First derivative: ∂w/∂μ
    # ∂w/∂μ = (θT/2) * [ρφ(θT) + φ(θT) * z/√(z² + (1 - ρ²))]
    #        = (θT/2) * φ(θT) * [ρ + z/√(z² + (1 - ρ²))]
    w_prime = (theta_T / 2) * phi_T * (rho + z / sqrt_term)
    
    # Second derivative: ∂²w/∂μ²
    # ∂²w/∂μ² = (θT/2) * φ(θT)² * (1 - ρ²) / (z² + (1 - ρ²))^(3/2)
    denominator = (z**2 + (1 - rho**2))**(3/2)
    w_double_prime = (theta_T / 2) * phi_T**2 * (1 - rho**2) / denominator
    
    return w, w_prime, w_double_prime


def compute_parametric_ssvi_time_derivative(
    mu_values: np.ndarray,
    T: float,
    rho: float,
    theta_inf: float,
    theta_0: float,
    kappa: float,
    p_coeffs: List[float],
    q_coeffs: List[float]
) -> np.ndarray:
    """
    Compute analytical time derivative ∂w/∂T for parametric SSVI.
    
    Mathematical derivation:
    w(μ, T) = (θT/2) * [1 + ρφ(θT)μ + √((φ(θT)μ + ρ)² + (1 - ρ²))]
    
    Using chain rule:
    ∂w/∂T = ∂w/∂θT * ∂θT/∂T + ∂w/∂φT * ∂φT/∂T
    
    Where:
    ∂θT/∂T = θ∞ + (θ0 - θ∞) * e^(-κT)
    ∂φT/∂T = ∂φ/∂θ * ∂θT/∂T
    
    Parameters:
    -----------
    mu_values : np.ndarray
        Log-moneyness values
    T : float
        Time to maturity
    rho : float
        Correlation parameter
    theta_inf : float
        Long-term variance level
    theta_0 : float
        Initial variance level
    kappa : float
        Mean reversion speed
    p_coeffs : List[float]
        Numerator coefficients for φ(θ)
    q_coeffs : List[float]
        Denominator coefficients for φ(θ)
        
    Returns:
    --------
    np.ndarray
        Time derivative ∂w/∂T
    """
    mu_values = np.asarray(mu_values)
    
    # Compute θ(T) and its derivative
    theta_T = compute_theta_T(np.array([T]), theta_inf, theta_0, kappa)[0]
    
    # ∂θT/∂T = θ∞ + (θ0 - θ∞) * e^(-κT)
    if abs(kappa) < 1e-10:
        # κ → 0 case: θT = θ0 * T, so ∂θT/∂T = θ0
        dtheta_dT = theta_0
    else:
        dtheta_dT = theta_inf + (theta_0 - theta_inf) * np.exp(-kappa * T)
    
    # Compute φ(θT) and its analytical derivative w.r.t. θ
    phi_T = compute_phi_rational(np.array([theta_T]), p_coeffs, q_coeffs)[0]
    
    # Compute ∂φ/∂θ analytically using the exact formula
    dphi_dtheta = compute_phi_rational_derivative(np.array([theta_T]), p_coeffs, q_coeffs)[0]
    
    # ∂φT/∂T = ∂φ/∂θ * ∂θT/∂T
    dphi_dT = dphi_dtheta * dtheta_dT
    
    # Intermediate variables
    z = phi_T * mu_values + rho
    sqrt_term = np.sqrt(z**2 + (1 - rho**2))
    
    # Compute ∂w/∂T using chain rule
    # w = (θT/2) * [1 + ρφT*μ + √((φT*μ + ρ)² + (1 - ρ²))]
    
    # ∂w/∂θT = (1/2) * [1 + ρφT*μ + √((φT*μ + ρ)² + (1 - ρ²))]
    dw_dtheta = 0.5 * (1 + rho * phi_T * mu_values + sqrt_term)
    
    # ∂w/∂φT = (θT/2) * [ρμ + μ * z/√(z² + (1 - ρ²))]
    #         = (θT/2) * μ * [ρ + z/√(z² + (1 - ρ²))]
    dw_dphi = (theta_T / 2) * mu_values * (rho + z / sqrt_term)
    
    # Total time derivative
    dw_dT = dw_dtheta * dtheta_dT + dw_dphi * dphi_dT
    
    return dw_dT


def compute_parametric_ssvi_all_derivatives(
    mu_values: np.ndarray,
    T: float,
    rho: float,
    theta_inf: float,
    theta_0: float,
    kappa: float,
    p_coeffs: List[float],
    q_coeffs: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all analytical derivatives for parametric SSVI.
    
    Returns total variance and all required derivatives for local volatility computation.
    
    Parameters:
    -----------
    mu_values : np.ndarray
        Log-moneyness values
    T : float
        Time to maturity
    rho : float
        Correlation parameter
    theta_inf : float
        Long-term variance level
    theta_0 : float
        Initial variance level
    kappa : float
        Mean reversion speed
    p_coeffs : List[float]
        Numerator coefficients for φ(θ)
    q_coeffs : List[float]
        Denominator coefficients for φ(θ)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (w, ∂w/∂μ, ∂²w/∂μ², ∂w/∂T)
    """
    # Compute spatial derivatives
    w, w_prime, w_double_prime = compute_parametric_ssvi_derivatives(
        mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    
    # Compute time derivative
    dw_dT = compute_parametric_ssvi_time_derivative(
        mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    
    return w, w_prime, w_double_prime, dw_dT


def compute_parametric_ssvi_local_volatility(
    mu_values: np.ndarray,
    T: float,
    rho: float,
    theta_inf: float,
    theta_0: float,
    kappa: float,
    p_coeffs: List[float],
    q_coeffs: List[float],
    r: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute local volatility from parametric SSVI using analytical derivatives.
    
    Uses the Dupire formula with all analytically computed derivatives.
    
    Parameters:
    -----------
    mu_values : np.ndarray
        Log-moneyness values
    T : float
        Time to maturity
    rho : float
        Correlation parameter
    theta_inf : float
        Long-term variance level
    theta_0 : float
        Initial variance level
    kappa : float
        Mean reversion speed
    p_coeffs : List[float]
        Numerator coefficients for φ(θ)
    q_coeffs : List[float]
        Denominator coefficients for φ(θ)
    r : float
        Risk-free rate
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
        (local_volatility, is_valid, diagnostics)
    """
    # Import here to avoid circular dependency
    from local_volatility import dupire_local_volatility_from_total_variance
    
    # Compute all derivatives analytically
    w, w_prime, w_double_prime, dw_dT = compute_parametric_ssvi_all_derivatives(
        mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    
    # Compute local volatility using Dupire formula
    local_vol, is_valid = dupire_local_volatility_from_total_variance(
        mu_values, T, w, w_prime, w_double_prime, dw_dT=dw_dT, r=r
    )
    
    # Prepare diagnostics
    diagnostics = {
        'total_variance': w,
        'first_derivative': w_prime,
        'second_derivative': w_double_prime,
        'time_derivative': dw_dT,
        'theta_T': compute_theta_T(np.array([T]), theta_inf, theta_0, kappa)[0],
        'phi_T': compute_phi_rational(
            compute_theta_T(np.array([T]), theta_inf, theta_0, kappa), 
            p_coeffs, q_coeffs
        )[0],
        'parameters': {
            'rho': rho, 'theta_inf': theta_inf, 'theta_0': theta_0, 
            'kappa': kappa, 'p_coeffs': p_coeffs, 'q_coeffs': q_coeffs,
            'T': T, 'r': r
        }
    }
    
    return local_vol, is_valid, diagnostics


def validate_parametric_ssvi_parameters(
    rho: float,
    theta_inf: float,
    theta_0: float,
    kappa: float,
    p_coeffs: List[float],
    q_coeffs: List[float]
) -> Tuple[bool, List[str]]:
    """
    Validate parametric SSVI parameters.
    
    Parameters:
    -----------
    rho : float
        Correlation parameter
    theta_inf : float
        Long-term variance level
    theta_0 : float
        Initial variance level
    kappa : float
        Mean reversion speed
    p_coeffs : List[float]
        Numerator coefficients [p0, p1, p2]
    q_coeffs : List[float]
        Denominator coefficients [q0, q1, q2]
        
    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list_of_violations)
    """
    violations = []
    
    # Check correlation parameter
    if not (-1 < rho < 1):
        violations.append(f"Correlation ρ = {rho} must be in (-1, 1)")
    
    # Check variance levels
    if theta_inf <= 0:
        violations.append(f"Long-term variance θ∞ = {theta_inf} must be positive")
    
    if theta_0 <= 0:
        violations.append(f"Initial variance θ0 = {theta_0} must be positive")
    
    # Check mean reversion speed
    if kappa < 0:
        violations.append(f"Mean reversion speed κ = {kappa} must be non-negative")
    
    # Check coefficient lengths
    if len(p_coeffs) != 3:
        violations.append(f"Numerator coefficients must have 3 elements, got {len(p_coeffs)}")
    
    if len(q_coeffs) != 3:
        violations.append(f"Denominator coefficients must have 3 elements, got {len(q_coeffs)}")
    
    # Check for reasonable φ(θ) behavior
    if len(p_coeffs) == 3 and len(q_coeffs) == 3:
        # Test φ(θ) at a few points to ensure it's positive and well-behaved
        test_theta_values = np.array([0.01, 0.1, 1.0, 10.0])
        try:
            phi_values = compute_phi_rational(test_theta_values, p_coeffs, q_coeffs)
            if np.any(phi_values <= 0):
                violations.append("φ(θ) must be positive for all relevant θ values")
            if np.any(~np.isfinite(phi_values)):
                violations.append("φ(θ) produces non-finite values")
        except Exception as e:
            violations.append(f"Error evaluating φ(θ): {str(e)}")
    
    is_valid = len(violations) == 0
    return is_valid, violations


def analyze_parametric_ssvi_properties(
    mu_values: np.ndarray,
    T_values: np.ndarray,
    rho: float,
    theta_inf: float,
    theta_0: float,
    kappa: float,
    p_coeffs: List[float],
    q_coeffs: List[float]
) -> Dict[str, Any]:
    """
    Analyze properties of the parametric SSVI model.
    
    Returns:
    --------
    Dict[str, Any]
        Analysis results including term structure properties and asymptotic behavior
    """
    analysis = {}
    
    # Parameter validation
    is_valid, violations = validate_parametric_ssvi_parameters(
        rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
    )
    analysis['parameter_validation'] = {'is_valid': is_valid, 'violations': violations}
    
    # Analyze θ(T) behavior
    theta_T_values = compute_theta_T(T_values, theta_inf, theta_0, kappa)
    analysis['theta_T'] = {
        'values': theta_T_values,
        'initial': theta_T_values[0] if len(theta_T_values) > 0 else None,
        'final': theta_T_values[-1] if len(theta_T_values) > 0 else None,
        'asymptotic_slope': theta_inf,
        'mean_reversion_strength': kappa
    }
    
    # Analyze φ(θ) behavior
    phi_values = compute_phi_rational(theta_T_values, p_coeffs, q_coeffs)
    analysis['phi_theta'] = {
        'values': phi_values,
        'min': np.min(phi_values) if len(phi_values) > 0 else None,
        'max': np.max(phi_values) if len(phi_values) > 0 else None,
        'range': np.max(phi_values) - np.min(phi_values) if len(phi_values) > 0 else None
    }
    
    # Compute surface properties
    if len(T_values) > 0 and len(mu_values) > 0:
        surface = compute_parametric_ssvi_surface(
            mu_values, T_values, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
        )
        analysis['surface'] = {
            'shape': surface.shape,
            'min_variance': np.min(surface),
            'max_variance': np.max(surface),
            'atm_term_structure': surface[:, len(mu_values)//2] if len(mu_values) > 0 else None
        }
    
    return analysis


def get_default_parametric_ssvi_parameters() -> Dict[str, Any]:
    """
    Get reasonable default parameters for the parametric SSVI model.
    
    Returns:
    --------
    Dict[str, Any]
        Default parameter values
    """
    return {
        'rho': -0.3,
        'theta_inf': 0.2,
        'theta_0': 0.1,
        'kappa': 2.0,
        'p_coeffs': [0.5, 0.1, 0.0],  # p0 + p1*θ + p2*θ²
        'q_coeffs': [1.0, 0.05, 0.0]  # q0 + q1*θ + q2*θ²
    }
