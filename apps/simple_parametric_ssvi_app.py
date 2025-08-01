#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Simple Parametric SSVI Interactive Plotter

A streamlined version of the parametric SSVI visualization focusing on the three
main plots: Implied Volatility, Local Volatility, and Risk-Neutral Density.

This version has a simpler layout and fewer parameters for easier use.

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings

from src.parametric_ssvi import (
    compute_parametric_ssvi_total_variance,
    compute_parametric_ssvi_all_derivatives,
    compute_parametric_ssvi_volatility_smile
)

# Try to import local volatility
try:
    from src.local_volatility import dupire_local_volatility_from_total_variance
    HAS_LOCAL_VOLATILITY = True
except ImportError:
    HAS_LOCAL_VOLATILITY = False


def create_simple_parametric_ssvi_app():
    """Create a simple interactive parametric SSVI application."""
    
    # Default parameters
    rho_init = -0.2
    theta_inf_init = 0.04
    theta_0_init = 0.09
    kappa_init = 2.0
    p0_init = 1.0
    p1_init = 0.2
    p2_init = -0.1
    q1_init = 0.1
    q2_init = 0.0
    
    # Display parameters
    maturities = [0.25, 0.5, 1.0, 2.0]
    mu_range = 2.0
    n_mu = 100
    r = 0.05
    
    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.35)
    
    # Function to update plots
    def update_plots(val=None):
        # Get current parameter values
        rho = slider_rho.val
        theta_inf = slider_theta_inf.val
        theta_0 = slider_theta_0.val
        kappa = slider_kappa.val
        p_coeffs = [slider_p0.val, slider_p1.val, slider_p2.val]
        q_coeffs = [1.0, slider_q1.val, slider_q2.val]
        
        mu_values = np.linspace(-mu_range, mu_range, n_mu)
        
        # Clear plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        try:
            # Plot 1: Implied Volatility
            colors = plt.cm.viridis(np.linspace(0, 1, len(maturities)))
            for T, color in zip(maturities, colors):
                sigma = compute_parametric_ssvi_volatility_smile(
                    mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
                )
                ax1.plot(mu_values, sigma, color=color, label=f'T={T:.2f}', linewidth=2)
            
            ax1.set_xlabel('Log-moneyness μ')
            ax1.set_ylabel('Implied Volatility')
            ax1.set_title('Implied Volatility Smiles')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Local Volatility Surface
            if HAS_LOCAL_VOLATILITY:
                mu_lv = np.linspace(-1.5, 1.5, 30)
                T_lv = np.linspace(0.1, 2.0, 25)
                MU, T_GRID = np.meshgrid(mu_lv, T_lv)
                LOCAL_VOL = np.zeros_like(MU)
                
                for i, T in enumerate(T_lv):
                    try:
                        w, w_prime, w_double_prime, dw_dT = compute_parametric_ssvi_all_derivatives(
                            mu_lv, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
                        )
                        local_vol, is_valid = dupire_local_volatility_from_total_variance(
                            mu_lv, T, w, w_prime, w_double_prime, dw_dT=dw_dT, r=r
                        )
                        LOCAL_VOL[i, :] = np.where(is_valid, local_vol, np.nan)
                    except:
                        LOCAL_VOL[i, :] = np.nan
                
                valid_mask = ~np.isnan(LOCAL_VOL)
                if np.any(valid_mask):
                    cs = ax2.contourf(MU, T_GRID, LOCAL_VOL, levels=15, cmap='plasma')
                    plt.colorbar(cs, ax=ax2, shrink=0.8)
                    
            else:
                ax2.text(0.5, 0.5, 'Local Volatility\\nNot Available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            
            ax2.set_xlabel('Log-moneyness μ')
            ax2.set_ylabel('Time to maturity T')
            ax2.set_title('Local Volatility Surface')
            
            # Plot 3: Risk-Neutral Density
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(maturities)))
            for T, color in zip(maturities, colors):
                try:
                    w, w_prime, w_double_prime, _ = compute_parametric_ssvi_all_derivatives(
                        mu_values, T, rho, theta_inf, theta_0, kappa, p_coeffs, q_coeffs
                    )
                    
                    # Simple density approximation
                    density = np.exp(-mu_values**2 / (2 * w)) / np.sqrt(2 * np.pi * w)
                    correction = 1 - mu_values * w_prime / 2 + w_double_prime / 2
                    density *= np.maximum(correction, 0.01)
                    density = density / np.trapz(density, mu_values)
                    
                    ax3.plot(mu_values, density, color=color, label=f'T={T:.2f}', linewidth=2)
                except:
                    continue
            
            ax3.set_xlabel('Log-moneyness μ')
            ax3.set_ylabel('Risk-Neutral Density')
            ax3.set_title('Risk-Neutral Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
        except Exception as e:
            # Show error message
            for ax in [ax1, ax2, ax3]:
                ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                       ha='center', va='center', transform=ax.transAxes, 
                       color='red', fontsize=10)
        
        fig.canvas.draw()
    
    # Create sliders
    slider_height = 0.02
    slider_spacing = 0.025
    start_y = 0.25
    
    # Row 1: Main SSVI parameters
    slider_rho = Slider(plt.axes([0.1, start_y, 0.15, slider_height]), 
                       'ρ', -0.95, 0.95, valinit=rho_init)
    slider_theta_inf = Slider(plt.axes([0.3, start_y, 0.15, slider_height]), 
                             'θ∞', 0.01, 0.5, valinit=theta_inf_init)
    slider_theta_0 = Slider(plt.axes([0.5, start_y, 0.15, slider_height]), 
                           'θ0', 0.01, 0.5, valinit=theta_0_init)
    slider_kappa = Slider(plt.axes([0.7, start_y, 0.15, slider_height]), 
                         'κ', 0.1, 10.0, valinit=kappa_init)
    
    # Row 2: Rational function numerator
    start_y -= slider_spacing
    slider_p0 = Slider(plt.axes([0.1, start_y, 0.15, slider_height]), 
                      'p₀', 0.1, 3.0, valinit=p0_init)
    slider_p1 = Slider(plt.axes([0.3, start_y, 0.15, slider_height]), 
                      'p₁', -2.0, 2.0, valinit=p1_init)
    slider_p2 = Slider(plt.axes([0.5, start_y, 0.15, slider_height]), 
                      'p₂', -2.0, 2.0, valinit=p2_init)
    
    # Row 3: Rational function denominator
    start_y -= slider_spacing
    slider_q1 = Slider(plt.axes([0.3, start_y, 0.15, slider_height]), 
                      'q₁', -2.0, 2.0, valinit=q1_init)
    slider_q2 = Slider(plt.axes([0.5, start_y, 0.15, slider_height]), 
                      'q₂', -2.0, 2.0, valinit=q2_init)
    
    # Connect sliders to update function
    sliders = [slider_rho, slider_theta_inf, slider_theta_0, slider_kappa,
               slider_p0, slider_p1, slider_p2, slider_q1, slider_q2]
    
    for slider in sliders:
        slider.on_changed(update_plots)
    
    # Initial plot
    update_plots()
    
    # Set window title
    fig.suptitle('Parametric SSVI: Implied Volatility, Local Volatility & Density', 
                fontsize=14, fontweight='bold')
    
    plt.show()


def main():
    """Main function."""
    print("Simple Parametric SSVI Interactive Plotter")
    print("=" * 42)
    print("Shows three plots side by side:")
    print("1. Implied Volatility smiles")
    print("2. Local Volatility surface")
    print("3. Risk-Neutral Density")
    print()
    print("Use sliders to adjust parameters in real-time.")
    print("Starting application...")
    
    create_simple_parametric_ssvi_app()


if __name__ == "__main__":
    main()
