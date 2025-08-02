#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Interactive Parametric SSVI Application with IV, LV, and Density Plots

This application provides an interactive visualization tool for the parametric SSVI model
with three main plots:
1. Implied Volatility (IV) - volatility smiles for different maturities
2. Local Volatility (LV) - local volatility surface using Dupire formula
3. Risk-Neutral Density - probability density functions

Features:
- Real-time parameter adjustment via sliders
- Multiple maturity visualization
- Parameter validation with error display
- Mathematical diagnostics
- Professional layout with clear labeling

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import warnings
from typing import List, Dict, Any, Tuple

# Import our parametric SSVI module
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parametric_ssvi import (
    compute_parametric_ssvi_total_variance,
    compute_parametric_ssvi_all_derivatives,
    compute_parametric_ssvi_volatility_smile,
    validate_parametric_ssvi_parameters,
    get_default_parametric_ssvi_parameters
)

# Try to import local volatility functionality
try:
    from src.local_volatility import dupire_local_volatility_from_total_variance
    HAS_LOCAL_VOLATILITY = True
except ImportError:
    HAS_LOCAL_VOLATILITY = False
    warnings.warn("Local volatility module not found. LV plot will be disabled.")


class ParametricSSVIVisualizationApp:
    """Interactive application for parametric SSVI visualization with IV, LV, and density."""
    
    def __init__(self):
        """Initialize the application."""
        self.setup_parameters()
        self.setup_figure()
        self.setup_controls()
        self.setup_display_options()
        
    def setup_parameters(self):
        """Initialize default parameter values."""
        defaults = get_default_parametric_ssvi_parameters()
        
        # Model parameters
        self.rho = defaults['rho']
        self.theta_inf = defaults['theta_inf']
        self.theta_0 = defaults['theta_0']
        self.kappa = defaults['kappa']
        
        # Rational function coefficients
        self.p_coeffs = defaults['p_coeffs'].copy()
        self.q_coeffs = defaults['q_coeffs'].copy()
        
        # Display parameters
        self.mu_range = 2.0
        self.n_mu = 100
        self.maturities = [0.25, 0.5, 1.0, 2.0]  # Multiple maturities for IV plot
        self.r = 0.05  # Risk-free rate for LV computation
        
    def setup_figure(self):
        """Set up the matplotlib figure and subplots."""
        plt.style.use('default')
        self.fig = plt.figure(figsize=(18, 12))
        
        # Create subplot layout with more space for plots
        # Top row: IV, LV, Density plots (moved lower to avoid title overlap)
        # Bottom row: Info panel and validation area
        
        self.ax_iv = plt.subplot2grid((4, 4), (1, 0), colspan=1)
        self.ax_lv = plt.subplot2grid((4, 4), (1, 1), colspan=1)
        self.ax_density = plt.subplot2grid((4, 4), (1, 2), colspan=1)
        self.ax_info = plt.subplot2grid((4, 4), (1, 3), colspan=1)
        
        # Parameter validation area
        self.ax_validation = plt.subplot2grid((4, 4), (2, 0), colspan=4)
        self.ax_validation.axis('off')
        
        # Add main title at the top with proper spacing
        self.fig.suptitle('Interactive Parametric SSVI Visualization', 
                         fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # Leave space for title and sliders
        
    def setup_controls(self):
        """Set up interactive parameter controls in two columns."""
        # Slider properties
        slider_props = {'facecolor': 'lightblue', 'alpha': 0.7}
        
        # Left column - Main SSVI parameters
        left_x = 0.20  # Moved from 0.05 to 0.20 for more centered position
        slider_width = 0.15
        slider_height = 0.025
        top_y = 0.30
        y_spacing = 0.035
        
        # Main SSVI parameters (left column)
        self.slider_rho = Slider(
            plt.axes([left_x, top_y, slider_width, slider_height]), 'ρ (correlation)', 
            -0.95, 0.95, valinit=self.rho, **slider_props
        )
        
        self.slider_theta_inf = Slider(
            plt.axes([left_x, top_y - y_spacing, slider_width, slider_height]), 'θ∞ (long-term)', 
            0.01, 1.0, valinit=self.theta_inf, **slider_props
        )
        
        self.slider_theta_0 = Slider(
            plt.axes([left_x, top_y - 2*y_spacing, slider_width, slider_height]), 'θ0 (initial)', 
            0.01, 1.0, valinit=self.theta_0, **slider_props
        )
        
        self.slider_kappa = Slider(
            plt.axes([left_x, top_y - 3*y_spacing, slider_width, slider_height]), 'κ (mean reversion)', 
            0.1, 10.0, valinit=self.kappa, **slider_props
        )
        
        # Numerator coefficients (left column continued)
        self.slider_p0 = Slider(
            plt.axes([left_x, top_y - 4*y_spacing, slider_width, slider_height]), 'p₀ (numerator)', 
            0.0, 3.0, valinit=self.p_coeffs[0], **slider_props
        )
        
        self.slider_p1 = Slider(
            plt.axes([left_x, top_y - 5*y_spacing, slider_width, slider_height]), 'p₁ (numerator)', 
            -2.0, 2.0, valinit=self.p_coeffs[1], **slider_props
        )
        
        # Right column - Rational function and display parameters
        right_x = 0.50  # Moved from 0.35 to 0.50 for better centering
        
        # Numerator and denominator coefficients (right column)
        self.slider_p2 = Slider(
            plt.axes([right_x, top_y, slider_width, slider_height]), 'p₂ (numerator)', 
            -2.0, 2.0, valinit=self.p_coeffs[2], **slider_props
        )
        
        self.slider_q1 = Slider(
            plt.axes([right_x, top_y - y_spacing, slider_width, slider_height]), 'q₁ (denominator)', 
            -2.0, 2.0, valinit=self.q_coeffs[1], **slider_props
        )
        
        self.slider_q2 = Slider(
            plt.axes([right_x, top_y - 2*y_spacing, slider_width, slider_height]), 'q₂ (denominator)', 
            -2.0, 2.0, valinit=self.q_coeffs[2], **slider_props
        )
        
        # Display controls (right column continued)
        self.slider_mu_range = Slider(
            plt.axes([right_x, top_y - 3*y_spacing, slider_width, slider_height]), 'μ range', 
            0.5, 3.0, valinit=self.mu_range, **slider_props
        )
        
        self.slider_r = Slider(
            plt.axes([right_x, top_y - 4*y_spacing, slider_width, slider_height]), 'r (risk-free rate)', 
            0.0, 0.20, valinit=self.r, **slider_props
        )
        
        # Reset button (right column)
        self.button_reset = Button(
            plt.axes([right_x, top_y - 5*y_spacing, slider_width, slider_height + 0.005]), 
            'Reset to Defaults',
            color='lightyellow'
        )
        
        # Connect callbacks
        self.connect_callbacks()
        
    def setup_display_options(self):
        """Set up display option controls."""
        # Info area setup
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        
    def connect_callbacks(self):
        """Connect slider and button callbacks."""
        sliders = [
            self.slider_rho, self.slider_theta_inf, self.slider_theta_0,
            self.slider_kappa, self.slider_p0, self.slider_p1, self.slider_p2,
            self.slider_q1, self.slider_q2, self.slider_mu_range, self.slider_r
        ]
        
        for slider in sliders:
            slider.on_changed(self.update_plots)
            
        self.button_reset.on_clicked(self.reset_parameters)
        
    def update_parameters(self):
        """Update parameters from sliders."""
        self.rho = self.slider_rho.val
        self.theta_inf = self.slider_theta_inf.val
        self.theta_0 = self.slider_theta_0.val
        self.kappa = self.slider_kappa.val
        
        self.p_coeffs = [
            self.slider_p0.val,
            self.slider_p1.val,
            self.slider_p2.val
        ]
        
        self.q_coeffs = [
            1.0,  # q0 is always 1
            self.slider_q1.val,
            self.slider_q2.val
        ]
        
        self.mu_range = self.slider_mu_range.val
        self.r = self.slider_r.val
        
    def validate_current_parameters(self) -> Tuple[bool, List[str]]:
        """Validate current parameter values."""
        return validate_parametric_ssvi_parameters(
            self.rho, self.theta_inf, self.theta_0, self.kappa,
            self.p_coeffs, self.q_coeffs
        )
        
    def update_plots(self, val=None):
        """Update all plots based on current parameters."""
        self.update_parameters()
        
        # Validate parameters
        is_valid, violations = self.validate_current_parameters()
        
        if not is_valid:
            self.show_parameter_violations(violations)
            return
            
        try:
            self.plot_implied_volatility()
            self.plot_local_volatility()
            self.plot_density()
            self.show_parameter_info()
            self.clear_validation_area()
            
        except Exception as e:
            self.show_error(f"Error in computation: {str(e)}")
            
        self.fig.canvas.draw()
        
    def plot_implied_volatility(self):
        """Plot implied volatility smiles for multiple maturities."""
        self.ax_iv.clear()
        
        mu_values = np.linspace(-self.mu_range, self.mu_range, self.n_mu)
        # Use consistent colormap across all plots
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.maturities)))
        
        for T, color in zip(self.maturities, colors):
            try:
                # Compute implied volatility
                sigma = compute_parametric_ssvi_volatility_smile(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
                
                self.ax_iv.plot(mu_values, sigma, color=color, 
                               label=f'T = {T:.2f}', linewidth=2)
                
            except Exception as e:
                # Skip this maturity if computation fails
                continue
                
        self.ax_iv.set_xlabel('Log-moneyness μ')
        self.ax_iv.set_ylabel('Implied Volatility σ')
        self.ax_iv.set_title('Implied Volatility Smiles')
        self.ax_iv.legend(fontsize=8)
        self.ax_iv.grid(True, alpha=0.3)
        
    def plot_local_volatility(self):
        """Plot local volatility curves for multiple maturities."""
        self.ax_lv.clear()
        
        if not HAS_LOCAL_VOLATILITY:
            self.ax_lv.text(0.5, 0.5, 'Local Volatility\nModule Not Available', 
                           ha='center', va='center', transform=self.ax_lv.transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            self.ax_lv.set_title('Local Volatility')
            return
            
        mu_values = np.linspace(-self.mu_range, self.mu_range, self.n_mu)
        # Use same colormap as implied volatility
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.maturities)))
        
        for T, color in zip(self.maturities, colors):
            try:
                # Get all derivatives
                w, w_prime, w_double_prime, dw_dT = compute_parametric_ssvi_all_derivatives(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
                
                # Compute local volatility using Dupire formula
                local_vol, is_valid = dupire_local_volatility_from_total_variance(
                    mu_values, T, w, w_prime, w_double_prime, dw_dT=dw_dT, r=self.r
                )
                
                # Only plot valid values
                valid_mu = mu_values[is_valid]
                valid_lv = local_vol[is_valid]
                
                if len(valid_mu) > 0:
                    self.ax_lv.plot(valid_mu, valid_lv, color=color, 
                                   label=f'T = {T:.2f}', linewidth=2)
                    
            except Exception as e:
                # Skip this maturity if computation fails
                continue
                
        self.ax_lv.set_xlabel('Log-moneyness μ')
        self.ax_lv.set_ylabel('Local Volatility σ_LV')
        self.ax_lv.set_title('Local Volatility Curves')
        self.ax_lv.legend(fontsize=8)
        self.ax_lv.grid(True, alpha=0.3)
        
    def plot_density(self):
        """Plot risk-neutral probability density."""
        self.ax_density.clear()
        
        mu_values = np.linspace(-self.mu_range, self.mu_range, 200)
        # Use same colormap as other plots for consistency
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.maturities)))
        
        for T, color in zip(self.maturities, colors):
            try:
                # Get derivatives for density computation
                w, w_prime, w_double_prime, _ = compute_parametric_ssvi_all_derivatives(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
                
                # Compute risk-neutral density using approximation
                # g(k) ≈ exp(-k²/(2w)) / sqrt(2πw) * (1 - k*w'/2 + w''/2)
                # This is a simplified approximation for visualization
                
                # Base Gaussian density
                density = np.exp(-mu_values**2 / (2 * w)) / np.sqrt(2 * np.pi * w)
                
                # Apply correction terms for better approximation
                correction = 1 - mu_values * w_prime / 2 + w_double_prime / 2
                density *= np.maximum(correction, 0.01)  # Ensure positive
                
                # Normalize to make it a proper density
                density = density / np.trapz(density, mu_values)
                
                self.ax_density.plot(mu_values, density, color=color, 
                                   label=f'T = {T:.2f}', linewidth=2)
                
            except Exception as e:
                # Skip this maturity if computation fails
                continue
                
        self.ax_density.set_xlabel('Log-moneyness μ')
        self.ax_density.set_ylabel('Risk-Neutral Density')
        self.ax_density.set_title('Risk-Neutral Density')
        self.ax_density.legend(fontsize=8)
        self.ax_density.grid(True, alpha=0.3)
        
    def show_parameter_info(self):
        """Display current parameter information and diagnostics."""
        self.ax_info.clear()
        
        # Compute some diagnostic information
        try:
            # Test at ATM for T=1
            mu_test = np.array([0.0])
            T_test = 1.0
            
            w_test, w_prime_test, w_double_prime_test, dw_dT_test = \
                compute_parametric_ssvi_all_derivatives(
                    mu_test, T_test, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
            
            sigma_atm = np.sqrt(w_test[0] / T_test)
            
            # Compute θ(T) values for different maturities
            from src.parametric_ssvi import compute_theta_T, compute_phi_rational
            
            theta_values = compute_theta_T(np.array(self.maturities), 
                                         self.theta_inf, self.theta_0, self.kappa)
            phi_values = compute_phi_rational(theta_values, self.p_coeffs, self.q_coeffs)
            
            # Check for flat smile condition (IV = LV)
            max_derivative = max(abs(w_prime_test[0]), abs(w_double_prime_test[0]))
            is_flat_smile = max_derivative < 1e-6
            flat_indicator = "✅ FLAT SMILE (IV≈LV)" if is_flat_smile else ""
            
            info_text = f"""Current Parameters:
            
SSVI Parameters:
ρ = {self.rho:.3f}
θ∞ = {self.theta_inf:.3f}
θ0 = {self.theta_0:.3f}
κ = {self.kappa:.3f}

Rational Function φ(θ):
p = [{self.p_coeffs[0]:.6f}, {self.p_coeffs[1]:.2f}, {self.p_coeffs[2]:.2f}]
q = [1.00, {self.q_coeffs[1]:.2f}, {self.q_coeffs[2]:.2f}]

Diagnostics (T=1, μ=0):
w(0,1) = {w_test[0]:.4f}
σ_ATM = {sigma_atm:.1%}
∂w/∂μ = {w_prime_test[0]:.8f}
∂²w/∂μ² = {w_double_prime_test[0]:.8f}
∂w/∂T = {dw_dT_test[0]:.4f}

{flat_indicator}

θ(T) Values:
T=0.25: {theta_values[0]:.4f}
T=0.50: {theta_values[1]:.4f}
T=1.00: {theta_values[2]:.4f}
T=2.00: {theta_values[3]:.4f}

φ(θT) Values:
T=0.25: {phi_values[0]:.4f}
T=0.50: {phi_values[1]:.4f}
T=1.00: {phi_values[2]:.4f}
T=2.00: {phi_values[3]:.4f}
"""
            
            self.ax_info.text(0.05, 0.90, info_text, transform=self.ax_info.transAxes,
                             verticalalignment='top', fontfamily='monospace', fontsize=8)
            
        except Exception as e:
            error_text = f"Info computation error:\n{str(e)}"
            self.ax_info.text(0.05, 0.90, error_text, transform=self.ax_info.transAxes,
                             verticalalignment='top', fontfamily='monospace', fontsize=8,
                             color='red')
        
        # Add title as text instead of using set_title to avoid axis display
        self.ax_info.text(0.5, 0.98, 'Model Diagnostics', transform=self.ax_info.transAxes,
                         ha='center', va='top', fontsize=10, fontweight='bold')
        self.ax_info.axis('off')  # Ensure axis is off
        
    def show_parameter_violations(self, violations: List[str]):
        """Show parameter validation errors."""
        self.ax_validation.clear()
        
        error_text = "⚠️  Parameter Violations:\n\n" + "\n".join(f"• {v}" for v in violations)
        
        self.ax_validation.text(0.05, 0.95, error_text, transform=self.ax_validation.transAxes,
                               verticalalignment='top', color='red', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        # Clear the main plots
        for ax in [self.ax_iv, self.ax_lv, self.ax_density]:
            ax.clear()
            ax.text(0.5, 0.5, 'Invalid Parameters', ha='center', va='center',
                   transform=ax.transAxes, color='red', fontsize=12)
            
    def clear_validation_area(self):
        """Clear the validation area when parameters are valid."""
        self.ax_validation.clear()
        self.ax_validation.axis('off')
        
    def show_error(self, error_msg: str):
        """Show computation error."""
        self.ax_validation.clear()
        self.ax_validation.text(0.05, 0.95, f"❌ Computation Error:\n{error_msg}", 
                               transform=self.ax_validation.transAxes,
                               verticalalignment='top', color='red', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
    def reset_parameters(self, event):
        """Reset all parameters to defaults."""
        defaults = get_default_parametric_ssvi_parameters()
        
        self.slider_rho.reset()
        self.slider_theta_inf.reset()
        self.slider_theta_0.reset()
        self.slider_kappa.reset()
        self.slider_p0.reset()
        self.slider_p1.reset()
        self.slider_p2.reset()
        self.slider_q1.reset()
        self.slider_q2.reset()
        self.slider_mu_range.reset()
        self.slider_r.reset()
        
        # Set to actual defaults
        self.slider_rho.set_val(defaults['rho'])
        self.slider_theta_inf.set_val(defaults['theta_inf'])
        self.slider_theta_0.set_val(defaults['theta_0'])
        self.slider_kappa.set_val(defaults['kappa'])
        self.slider_p0.set_val(defaults['p_coeffs'][0])
        self.slider_p1.set_val(defaults['p_coeffs'][1])
        self.slider_p2.set_val(defaults['p_coeffs'][2])
        self.slider_q1.set_val(defaults['q_coeffs'][1])
        self.slider_q2.set_val(defaults['q_coeffs'][2])
        
        self.update_plots()
        
    def run(self):
        """Run the interactive application."""
        # Initial plot
        self.update_plots()
        
        # Add title
        self.fig.suptitle('Interactive Parametric SSVI: Implied Volatility, Local Volatility & Density', 
                         fontsize=14, fontweight='bold')
        
        # Show the plot
        plt.show()


def main():
    """Main function to run the parametric SSVI visualization application."""
    print("Interactive Parametric SSVI Visualization")
    print("=" * 45)
    print("Features:")
    print("- Implied Volatility smiles for multiple maturities")
    print("- Local Volatility surface using analytical derivatives")
    print("- Risk-Neutral Density functions")
    print("- Real-time parameter adjustment with sliders")
    print("- Parameter validation and diagnostics")
    print()
    print("Parameters:")
    print("- ρ: Correlation parameter")
    print("- θ∞, θ0, κ: Time-dependent variance parameters")
    print("- p0, p1, p2: Rational function numerator coefficients")
    print("- q1, q2: Rational function denominator coefficients (q0=1)")
    print("- μ range: Log-moneyness display range")
    print("- r: Risk-free rate for local volatility")
    print()
    print("Starting application...")
    
    app = ParametricSSVIVisualizationApp()
    app.run()


if __name__ == "__main__":
    main()
