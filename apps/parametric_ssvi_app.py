#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Interactive Parametric SSVI Application

This application provides interactive visualization of the extended parametric SSVI model:
- Time-dependent variance level θ(T) with mean reversion
- Rational function φ(θ) for skew dynamics  
- Local volatility computation
- Risk-neutral density analysis
- Complete parameter sensitivity analysis

Features:
1. 3D parametric SSVI surface visualization
2. Time structure analysis (θ(T) and φ(θ))
3. Volatility smile comparison (implied vs local)
4. Risk-neutral density with arbitrage detection
5. Interactive parameter sliders for all model parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import warnings

from src.parametric_ssvi import (
    compute_parametric_ssvi_surface,
    compute_parametric_ssvi_volatility_smile,
    compute_parametric_ssvi_derivatives,
    compute_theta_T,
    compute_phi_rational,
    validate_parametric_ssvi_parameters,
    analyze_parametric_ssvi_properties,
    get_default_parametric_ssvi_parameters
)

from src.local_volatility import dupire_local_volatility_from_total_variance
from src.density_analysis import compute_svi_risk_neutral_density, verify_density_properties

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ParametricSSVIApp:
    """Interactive parametric SSVI visualization application."""
    
    def __init__(self):
        """Initialize the application."""
        self.setup_parameters()
        self.setup_grids()
        
    def setup_parameters(self):
        """Setup default parameters."""
        defaults = get_default_parametric_ssvi_parameters()
        self.rho = defaults['rho']
        self.theta_inf = defaults['theta_inf']
        self.theta_0 = defaults['theta_0']
        self.kappa = defaults['kappa']
        self.p_coeffs = defaults['p_coeffs'].copy()
        self.q_coeffs = defaults['q_coeffs'].copy()
        
        # Analysis parameters
        self.r = 0.02  # Risk-free rate for local volatility
        
    def setup_grids(self):
        """Setup computation grids."""
        # Moneyness grid
        self.mu_values = np.linspace(-3.0, 3.0, 100)
        
        # Time grid
        self.T_values = np.linspace(0.1, 5.0, 20)
        
        # Fixed time for smile analysis
        self.T_fixed = 1.0
        
    def run_3d_surface_visualization(self):
        """Run 3D surface visualization with interactive controls."""
        print("Launching 3D Parametric SSVI Surface Visualization...")
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initial surface computation
        surface = self.compute_surface()
        MU, T = np.meshgrid(self.mu_values, self.T_values)
        
        # Plot surface
        surf = ax.plot_surface(MU, T, surface, cmap='viridis', alpha=0.8)
        ax.set_xlabel('Log-Moneyness μ')
        ax.set_ylabel('Time to Maturity T')
        ax.set_zlabel('Total Variance w(μ,T)')
        ax.set_title('Parametric SSVI Total Variance Surface')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Create sliders
        plt.subplots_adjust(bottom=0.35)
        self.create_sliders_3d(fig, ax, surf)
        
        # Display parameter info
        self.display_parameter_info()
        
        plt.show()
        
    def run_time_structure_analysis(self):
        """Run time structure analysis showing θ(T) and φ(θ) evolution."""
        print("Launching Time Structure Analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot initial curves
        self.plot_time_structure(ax1, ax2, ax3, ax4)
        
        # Create sliders
        plt.subplots_adjust(bottom=0.3)
        self.create_sliders_time_structure(fig, ax1, ax2, ax3, ax4)
        
        plt.show()
        
    def run_volatility_analysis(self):
        """Run volatility analysis with implied vs local volatility."""
        print("Launching Volatility Analysis (Implied vs Local)...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot initial analysis
        self.plot_volatility_analysis(ax1, ax2, ax3, ax4)
        
        # Create sliders
        plt.subplots_adjust(bottom=0.3)
        self.create_sliders_volatility(fig, ax1, ax2, ax3, ax4)
        
        plt.show()
        
    def run_density_analysis(self):
        """Run risk-neutral density analysis."""
        print("Launching Risk-Neutral Density Analysis...")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot initial analysis
        self.plot_density_analysis(ax1, ax2, ax3)
        
        # Create sliders
        plt.subplots_adjust(bottom=0.25)
        self.create_sliders_density(fig, ax1, ax2, ax3)
        
        plt.show()
        
    def compute_surface(self):
        """Compute parametric SSVI surface."""
        return compute_parametric_ssvi_surface(
            self.mu_values, self.T_values, self.rho, 
            self.theta_inf, self.theta_0, self.kappa,
            self.p_coeffs, self.q_coeffs
        )
        
    def compute_smile_and_derivatives(self):
        """Compute volatility smile and derivatives for local volatility."""
        # Total variance and derivatives
        w, w_prime, w_double_prime = compute_parametric_ssvi_derivatives(
            self.mu_values, self.T_fixed, self.rho,
            self.theta_inf, self.theta_0, self.kappa,
            self.p_coeffs, self.q_coeffs
        )
        
        # Implied volatility
        implied_vol = np.sqrt(np.maximum(w / self.T_fixed, 1e-12))
        
        # Local volatility (simplified approach: dw_dT = w/T)
        dw_dT = w / self.T_fixed
        local_vol, is_valid = dupire_local_volatility_from_total_variance(
            self.mu_values, self.T_fixed, w, w_prime, w_double_prime, 
            dw_dT=dw_dT, r=self.r
        )
        
        return w, implied_vol, local_vol, is_valid
        
    def plot_time_structure(self, ax1, ax2, ax3, ax4):
        """Plot time structure analysis."""
        # Clear axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
            
        # Compute θ(T) and φ(θ)
        theta_T_values = compute_theta_T(self.T_values, self.theta_inf, self.theta_0, self.kappa)
        phi_values = compute_phi_rational(theta_T_values, self.p_coeffs, self.q_coeffs)
        
        # Plot θ(T)
        ax1.plot(self.T_values, theta_T_values, 'b-', linewidth=2, label='θ(T)')
        ax1.axhline(y=self.theta_inf * self.T_values[-1], color='r', linestyle='--', 
                   label=f'θ∞T (asymptote)')
        ax1.set_xlabel('Time T')
        ax1.set_ylabel('θ(T)')
        ax1.set_title('Time-Dependent Variance Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot φ(θ)
        theta_range = np.linspace(0.01, max(theta_T_values) * 1.2, 100)
        phi_range = compute_phi_rational(theta_range, self.p_coeffs, self.q_coeffs)
        ax2.plot(theta_range, phi_range, 'g-', linewidth=2, label='φ(θ)')
        ax2.scatter(theta_T_values, phi_values, color='red', s=30, 
                   label='φ(θ(T)) for T grid', zorder=5)
        ax2.set_xlabel('θ')
        ax2.set_ylabel('φ(θ)')
        ax2.set_title('Rational Skew Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot ATM variance term structure
        atm_idx = len(self.mu_values) // 2
        surface = self.compute_surface()
        atm_variance = surface[:, atm_idx]
        ax3.plot(self.T_values, atm_variance, 'purple', linewidth=2, label='ATM Total Variance')
        ax3.set_xlabel('Time T')
        ax3.set_ylabel('w(0, T)')
        ax3.set_title('ATM Variance Term Structure')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot parameter values as text
        ax4.text(0.1, 0.9, f'ρ = {self.rho:.3f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.8, f'θ∞ = {self.theta_inf:.3f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.7, f'θ0 = {self.theta_0:.3f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f'κ = {self.kappa:.3f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.5, f'p = [{self.p_coeffs[0]:.2f}, {self.p_coeffs[1]:.2f}, {self.p_coeffs[2]:.2f}]', 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f'q = [{self.q_coeffs[0]:.2f}, {self.q_coeffs[1]:.2f}, {self.q_coeffs[2]:.2f}]', 
                transform=ax4.transAxes, fontsize=12)
        
        # Validation
        is_valid, violations = validate_parametric_ssvi_parameters(
            self.rho, self.theta_inf, self.theta_0, self.kappa, 
            self.p_coeffs, self.q_coeffs
        )
        status_color = 'green' if is_valid else 'red'
        status_text = 'Valid' if is_valid else 'Invalid'
        ax4.text(0.1, 0.2, f'Status: {status_text}', transform=ax4.transAxes, 
                fontsize=14, color=status_color, weight='bold')
        
        if violations:
            ax4.text(0.1, 0.1, f'Issues: {len(violations)}', transform=ax4.transAxes, 
                    fontsize=10, color='red')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Model Parameters')
        ax4.axis('off')
        
    def plot_volatility_analysis(self, ax1, ax2, ax3, ax4):
        """Plot volatility analysis."""
        # Clear axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
            
        try:
            # Compute volatilities
            w, implied_vol, local_vol, is_valid = self.compute_smile_and_derivatives()
            
            # Apply validity mask
            local_vol_masked = np.where(is_valid, local_vol, np.nan)
            
            # Plot implied and local volatility
            ax1.plot(self.mu_values, implied_vol, 'b-', linewidth=2, label='Implied Volatility', alpha=0.8)
            ax1.plot(self.mu_values, local_vol_masked, 'r-', linewidth=2, label='Local Volatility', alpha=0.8)
            
            # Mark invalid regions
            if np.any(~is_valid):
                ax1.scatter(self.mu_values[~is_valid], np.full(np.sum(~is_valid), np.nan),
                           color='red', s=20, marker='x', label='Invalid LV', zorder=5)
            
            ax1.set_xlabel('Log-Moneyness μ')
            ax1.set_ylabel('Volatility')
            ax1.set_title(f'Implied vs Local Volatility (T={self.T_fixed})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot volatility ratio
            valid_mask = is_valid & (implied_vol > 1e-6)
            if np.any(valid_mask):
                ratio = local_vol[valid_mask] / implied_vol[valid_mask]
                ax2.plot(self.mu_values[valid_mask], ratio, 'g-', linewidth=2, label='LV/IV Ratio')
                ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='LV = IV')
                ax2.set_xlabel('Log-Moneyness μ')
                ax2.set_ylabel('Local/Implied Ratio')
                ax2.set_title('Volatility Ratio Analysis')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Plot total variance and derivatives
            w, w_prime, w_double_prime = compute_parametric_ssvi_derivatives(
                self.mu_values, self.T_fixed, self.rho,
                self.theta_inf, self.theta_0, self.kappa,
                self.p_coeffs, self.q_coeffs
            )
            
            ax3.plot(self.mu_values, w, 'b-', linewidth=2, label='w(μ,T)')
            ax3.plot(self.mu_values, w_prime, 'g-', linewidth=2, label="w'(μ,T)")
            ax3.plot(self.mu_values, w_double_prime, 'r-', linewidth=2, label='w"(μ,T)')
            ax3.set_xlabel('Log-Moneyness μ')
            ax3.set_ylabel('Derivatives')
            ax3.set_title('Total Variance and Derivatives')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Statistics
            valid_count = np.sum(is_valid)
            total_count = len(is_valid)
            
            ax4.text(0.1, 0.9, f'Time: T = {self.T_fixed}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.8, f'Valid LV points: {valid_count}/{total_count}', transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f'IV range: {np.min(implied_vol):.4f} - {np.max(implied_vol):.4f}', 
                    transform=ax4.transAxes, fontsize=12)
            
            if valid_count > 0:
                valid_lv = local_vol[is_valid]
                ax4.text(0.1, 0.6, f'LV range: {np.min(valid_lv):.4f} - {np.max(valid_lv):.4f}', 
                        transform=ax4.transAxes, fontsize=12)
                
                if np.any(valid_mask):
                    ax4.text(0.1, 0.5, f'Ratio range: {np.min(ratio):.4f} - {np.max(ratio):.4f}', 
                            transform=ax4.transAxes, fontsize=12)
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('Statistics')
            ax4.axis('off')
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)}', transform=ax1.transAxes, 
                    ha='center', va='center', fontsize=12, color='red')
            ax1.set_title('Volatility Analysis - Error')
            
    def plot_density_analysis(self, ax1, ax2, ax3):
        """Plot risk-neutral density analysis."""
        # Clear axes
        for ax in [ax1, ax2, ax3]:
            ax.clear()
            
        try:
            # Compute volatility smile
            implied_vol = compute_parametric_ssvi_volatility_smile(
                self.mu_values, self.T_fixed, self.rho,
                self.theta_inf, self.theta_0, self.kappa,
                self.p_coeffs, self.q_coeffs
            )
            
            # Compute density using SVI approximation (convert parameters)
            # This is approximate - we're using the SVI density formula
            # For exact density, we'd need the parametric SSVI density formula
            w = compute_parametric_ssvi_derivatives(
                self.mu_values, self.T_fixed, self.rho,
                self.theta_inf, self.theta_0, self.kappa,
                self.p_coeffs, self.q_coeffs
            )[0]
            
            # Approximate SVI parameters for density calculation
            # This is a simplification - real implementation would derive exact density
            atm_var = w[len(w)//2]
            a_approx = atm_var * 0.8
            b_approx = 0.3
            rho_approx = self.rho
            m_approx = 0.0
            sigma_approx = 0.3
            
            density = compute_svi_risk_neutral_density(
                self.mu_values, self.T_fixed, a_approx, b_approx, 
                rho_approx, m_approx, sigma_approx
            )
            
            # Verify density properties
            verification = verify_density_properties(self.mu_values, density)
            
            # Plot volatility smile
            ax1.plot(self.mu_values, implied_vol, 'b-', linewidth=2, label='Parametric SSVI Smile')
            ax1.set_xlabel('Log-Moneyness μ')
            ax1.set_ylabel('Implied Volatility')
            ax1.set_title(f'Parametric SSVI Volatility Smile (T={self.T_fixed})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot density
            ax2.plot(self.mu_values, density, 'g-', linewidth=2, label='Risk-Neutral Density')
            ax2.fill_between(self.mu_values, 0, density, alpha=0.3, color='green')
            ax2.set_xlabel('Log-Moneyness μ')
            ax2.set_ylabel('Density')
            ax2.set_title('Risk-Neutral Density (Approximate)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Display verification results
            ax3.text(0.1, 0.9, f"All non-negative: {verification['all_non_negative']}", 
                    transform=ax3.transAxes, fontsize=12)
            ax3.text(0.1, 0.8, f"Total probability: {verification['total_probability']:.6f}", 
                    transform=ax3.transAxes, fontsize=12)
            ax3.text(0.1, 0.7, f"Expected μ: {verification['expected_log_moneyness']:.6f}", 
                    transform=ax3.transAxes, fontsize=12)
            ax3.text(0.1, 0.6, f"Contains inf: {verification['contains_inf']}", 
                    transform=ax3.transAxes, fontsize=12)
            ax3.text(0.1, 0.5, f"Contains NaN: {verification['contains_nan']}", 
                    transform=ax3.transAxes, fontsize=12)
            ax3.text(0.1, 0.3, "Note: Density is approximate", 
                    transform=ax3.transAxes, fontsize=10, style='italic')
            ax3.text(0.1, 0.2, "Exact parametric SSVI density", 
                    transform=ax3.transAxes, fontsize=10, style='italic')
            ax3.text(0.1, 0.1, "requires specialized implementation", 
                    transform=ax3.transAxes, fontsize=10, style='italic')
            
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_title('Density Verification')
            ax3.axis('off')
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)}', transform=ax1.transAxes, 
                    ha='center', va='center', fontsize=12, color='red')
            ax1.set_title('Density Analysis - Error')
            
    def create_sliders_3d(self, fig, ax, surf):
        """Create sliders for 3D surface visualization."""
        # Slider positions
        slider_height = 0.03
        slider_spacing = 0.04
        slider_left = 0.2
        slider_width = 0.6
        
        # Create sliders
        ax_rho = plt.axes([slider_left, 0.02, slider_width, slider_height])
        self.slider_rho = Slider(ax_rho, 'ρ', -0.99, 0.99, valinit=self.rho, valfmt='%.3f')
        
        ax_theta_inf = plt.axes([slider_left, 0.02 + slider_spacing, slider_width, slider_height])
        self.slider_theta_inf = Slider(ax_theta_inf, 'θ∞', 0.01, 1.0, valinit=self.theta_inf, valfmt='%.3f')
        
        ax_theta_0 = plt.axes([slider_left, 0.02 + 2*slider_spacing, slider_width, slider_height])
        self.slider_theta_0 = Slider(ax_theta_0, 'θ0', 0.01, 1.0, valinit=self.theta_0, valfmt='%.3f')
        
        ax_kappa = plt.axes([slider_left, 0.02 + 3*slider_spacing, slider_width, slider_height])
        self.slider_kappa = Slider(ax_kappa, 'κ', 0.1, 10.0, valinit=self.kappa, valfmt='%.3f')
        
        ax_p0 = plt.axes([slider_left, 0.02 + 4*slider_spacing, slider_width, slider_height])
        self.slider_p0 = Slider(ax_p0, 'p0', 0.1, 2.0, valinit=self.p_coeffs[0], valfmt='%.3f')
        
        ax_p1 = plt.axes([slider_left, 0.02 + 5*slider_spacing, slider_width, slider_height])
        self.slider_p1 = Slider(ax_p1, 'p1', -1.0, 1.0, valinit=self.p_coeffs[1], valfmt='%.3f')
        
        # Update function
        def update_surface(val):
            self.rho = self.slider_rho.val
            self.theta_inf = self.slider_theta_inf.val
            self.theta_0 = self.slider_theta_0.val
            self.kappa = self.slider_kappa.val
            self.p_coeffs[0] = self.slider_p0.val
            self.p_coeffs[1] = self.slider_p1.val
            
            # Recompute surface
            try:
                new_surface = self.compute_surface()
                ax.clear()
                MU, T = np.meshgrid(self.mu_values, self.T_values)
                surf = ax.plot_surface(MU, T, new_surface, cmap='viridis', alpha=0.8)
                ax.set_xlabel('Log-Moneyness μ')
                ax.set_ylabel('Time to Maturity T')
                ax.set_zlabel('Total Variance w(μ,T)')
                ax.set_title('Parametric SSVI Total Variance Surface')
                fig.canvas.draw()
            except Exception as e:
                print(f"Error updating surface: {e}")
        
        # Connect sliders
        self.slider_rho.on_changed(update_surface)
        self.slider_theta_inf.on_changed(update_surface)
        self.slider_theta_0.on_changed(update_surface)
        self.slider_kappa.on_changed(update_surface)
        self.slider_p0.on_changed(update_surface)
        self.slider_p1.on_changed(update_surface)
        
    def create_sliders_time_structure(self, fig, ax1, ax2, ax3, ax4):
        """Create sliders for time structure analysis."""
        # Similar slider creation pattern as above
        slider_height = 0.03
        slider_spacing = 0.04
        slider_left = 0.2
        slider_width = 0.6
        
        # Create sliders
        ax_rho = plt.axes([slider_left, 0.02, slider_width, slider_height])
        self.slider_rho = Slider(ax_rho, 'ρ', -0.99, 0.99, valinit=self.rho, valfmt='%.3f')
        
        ax_theta_inf = plt.axes([slider_left, 0.02 + slider_spacing, slider_width, slider_height])
        self.slider_theta_inf = Slider(ax_theta_inf, 'θ∞', 0.01, 1.0, valinit=self.theta_inf, valfmt='%.3f')
        
        ax_theta_0 = plt.axes([slider_left, 0.02 + 2*slider_spacing, slider_width, slider_height])
        self.slider_theta_0 = Slider(ax_theta_0, 'θ0', 0.01, 1.0, valinit=self.theta_0, valfmt='%.3f')
        
        ax_kappa = plt.axes([slider_left, 0.02 + 3*slider_spacing, slider_width, slider_height])
        self.slider_kappa = Slider(ax_kappa, 'κ', 0.1, 10.0, valinit=self.kappa, valfmt='%.3f')
        
        def update_plots(val):
            self.rho = self.slider_rho.val
            self.theta_inf = self.slider_theta_inf.val
            self.theta_0 = self.slider_theta_0.val
            self.kappa = self.slider_kappa.val
            
            self.plot_time_structure(ax1, ax2, ax3, ax4)
            fig.canvas.draw()
        
        self.slider_rho.on_changed(update_plots)
        self.slider_theta_inf.on_changed(update_plots)
        self.slider_theta_0.on_changed(update_plots)
        self.slider_kappa.on_changed(update_plots)
        
    def create_sliders_volatility(self, fig, ax1, ax2, ax3, ax4):
        """Create sliders for volatility analysis."""
        # Similar pattern as other slider functions
        slider_height = 0.03
        slider_spacing = 0.04
        slider_left = 0.2
        slider_width = 0.6
        
        ax_rho = plt.axes([slider_left, 0.02, slider_width, slider_height])
        self.slider_rho = Slider(ax_rho, 'ρ', -0.99, 0.99, valinit=self.rho, valfmt='%.3f')
        
        ax_T = plt.axes([slider_left, 0.02 + slider_spacing, slider_width, slider_height])
        self.slider_T = Slider(ax_T, 'T', 0.1, 5.0, valinit=self.T_fixed, valfmt='%.2f')
        
        def update_plots(val):
            self.rho = self.slider_rho.val
            self.T_fixed = self.slider_T.val
            
            self.plot_volatility_analysis(ax1, ax2, ax3, ax4)
            fig.canvas.draw()
        
        self.slider_rho.on_changed(update_plots)
        self.slider_T.on_changed(update_plots)
        
    def create_sliders_density(self, fig, ax1, ax2, ax3):
        """Create sliders for density analysis."""
        slider_height = 0.03
        slider_spacing = 0.04
        slider_left = 0.2
        slider_width = 0.6
        
        ax_T = plt.axes([slider_left, 0.02, slider_width, slider_height])
        self.slider_T = Slider(ax_T, 'T', 0.1, 5.0, valinit=self.T_fixed, valfmt='%.2f')
        
        def update_plots(val):
            self.T_fixed = self.slider_T.val
            
            self.plot_density_analysis(ax1, ax2, ax3)
            fig.canvas.draw()
        
        self.slider_T.on_changed(update_plots)
        
    def display_parameter_info(self):
        """Display current parameter information."""
        print("\n=== Current Parametric SSVI Parameters ===")
        print(f"ρ (correlation): {self.rho:.3f}")
        print(f"θ∞ (long-term variance): {self.theta_inf:.3f}")
        print(f"θ0 (initial variance): {self.theta_0:.3f}")
        print(f"κ (mean reversion): {self.kappa:.3f}")
        print(f"p coefficients: [{self.p_coeffs[0]:.3f}, {self.p_coeffs[1]:.3f}, {self.p_coeffs[2]:.3f}]")
        print(f"q coefficients: [{self.q_coeffs[0]:.3f}, {self.q_coeffs[1]:.3f}, {self.q_coeffs[2]:.3f}]")
        
        # Validation
        is_valid, violations = validate_parametric_ssvi_parameters(
            self.rho, self.theta_inf, self.theta_0, self.kappa, 
            self.p_coeffs, self.q_coeffs
        )
        print(f"\nParameter validation: {'PASSED' if is_valid else 'FAILED'}")
        if violations:
            for violation in violations:
                print(f"  - {violation}")
                
        print("============================================\n")


def main():
    """Main application entry point."""
    print("Parametric SSVI Interactive Application")
    print("========================================")
    print("This application provides comprehensive analysis of the extended parametric SSVI model.")
    print("\nFeatures:")
    print("1. 3D surface visualization with interactive parameter control")
    print("2. Time structure analysis (θ(T) and φ(θ) dynamics)")
    print("3. Volatility analysis (implied vs local volatility)")
    print("4. Risk-neutral density analysis (approximate)")
    print("\nAll visualizations include interactive parameter sliders.")
    
    while True:
        print("\nSelect analysis mode:")
        print("1. 3D Parametric SSVI Surface")
        print("2. Time Structure Analysis")
        print("3. Volatility Analysis (Implied vs Local)")
        print("4. Risk-Neutral Density Analysis")
        print("5. Exit")
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                app = ParametricSSVIApp()
                app.run_3d_surface_visualization()
            elif choice == '2':
                app = ParametricSSVIApp()
                app.run_time_structure_analysis()
            elif choice == '3':
                app = ParametricSSVIApp()
                app.run_volatility_analysis()
            elif choice == '4':
                app = ParametricSSVIApp()
                app.run_density_analysis()
            elif choice == '5':
                print("Exiting application...")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting application...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
