#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Enhanced Parametric SSVI Interactive Application with Analytical Derivatives

This application provides an enhanced interactive visualization tool for the parametric SSVI model
with analytical derivative computation for improved performance and accuracy.

Features:
- All visualization modes from the original application
- Analytical derivative computation for superior performance
- Local volatility computation
- Performance benchmarking capabilities
- Enhanced parameter validation and diagnostics

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import time
from typing import List, Dict, Any, Tuple
import warnings

# Import our parametric SSVI module
from src.parametric_ssvi import (
    compute_parametric_ssvi_total_variance,
    compute_parametric_ssvi_derivatives,
    compute_parametric_ssvi_all_derivatives,
    compute_parametric_ssvi_time_derivative,
    compute_parametric_ssvi_local_volatility,
    validate_parametric_ssvi_parameters,
    analyze_parametric_ssvi_properties
)

# Import local volatility module for comparison
try:
    from src.local_volatility import dupire_local_volatility_from_total_variance
    HAS_LOCAL_VOLATILITY = True
except ImportError:
    HAS_LOCAL_VOLATILITY = False
    warnings.warn("Local volatility module not found. Some features may be unavailable.")


class EnhancedParametricSSVIApp:
    """Enhanced interactive application for parametric SSVI analysis."""
    
    def __init__(self):
        """Initialize the enhanced application."""
        self.setup_parameters()
        self.setup_figure()
        self.setup_controls()
        self.current_mode = "3D Surface"
        self.analytical_mode = True  # Use analytical derivatives by default
        self.benchmark_data = {}
        
    def setup_parameters(self):
        """Initialize default parameter values."""
        # Model parameters
        self.rho = 0.1
        self.theta_inf = 0.04
        self.theta_0 = 0.09
        self.kappa = 2.0
        
        # Rational function coefficients
        self.p_coeffs = [1.0, 0.2, -0.1]
        self.q_coeffs = [1.0, 0.1, 0.0]
        
        # Grid parameters
        self.mu_range = 2.0
        self.T_min, self.T_max = 0.1, 3.0
        self.n_mu = 50
        self.n_T = 30
        
        # Risk-free rate for local volatility
        self.r = 0.05
        
    def setup_figure(self):
        """Set up the matplotlib figure and subplots."""
        plt.style.use('default')
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main plot area
        self.ax_main = plt.subplot2grid((3, 4), (0, 1), colspan=3, rowspan=2)
        
        # Controls area
        self.ax_controls = plt.subplot2grid((3, 4), (0, 0), rowspan=3)
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis('off')
        
        # Info area
        self.ax_info = plt.subplot2grid((3, 4), (2, 1), colspan=3)
        self.ax_info.axis('off')
        
        plt.tight_layout()
        
    def setup_controls(self):
        """Set up interactive controls."""
        # Parameter sliders
        slider_props = {'facecolor': 'lightblue', 'alpha': 0.7}
        
        # Main parameters
        self.slider_rho = Slider(
            plt.axes([0.05, 0.85, 0.15, 0.03]), 'ρ', -0.95, 0.95, 
            valinit=self.rho, **slider_props
        )
        
        self.slider_theta_inf = Slider(
            plt.axes([0.05, 0.80, 0.15, 0.03]), 'θ∞', 0.01, 0.5, 
            valinit=self.theta_inf, **slider_props
        )
        
        self.slider_theta_0 = Slider(
            plt.axes([0.05, 0.75, 0.15, 0.03]), 'θ0', 0.01, 0.5, 
            valinit=self.theta_0, **slider_props
        )
        
        self.slider_kappa = Slider(
            plt.axes([0.05, 0.70, 0.15, 0.03]), 'κ', 0.1, 10.0, 
            valinit=self.kappa, **slider_props
        )
        
        # Rational function coefficients
        self.slider_p0 = Slider(
            plt.axes([0.05, 0.60, 0.15, 0.03]), 'p₀', 0.1, 3.0, 
            valinit=self.p_coeffs[0], **slider_props
        )
        
        self.slider_p1 = Slider(
            plt.axes([0.05, 0.55, 0.15, 0.03]), 'p₁', -1.0, 1.0, 
            valinit=self.p_coeffs[1], **slider_props
        )
        
        self.slider_p2 = Slider(
            plt.axes([0.05, 0.50, 0.15, 0.03]), 'p₂', -1.0, 1.0, 
            valinit=self.p_coeffs[2], **slider_props
        )
        
        self.slider_q1 = Slider(
            plt.axes([0.05, 0.40, 0.15, 0.03]), 'q₁', -1.0, 1.0, 
            valinit=self.q_coeffs[1], **slider_props
        )
        
        self.slider_q2 = Slider(
            plt.axes([0.05, 0.35, 0.15, 0.03]), 'q₂', -1.0, 1.0, 
            valinit=self.q_coeffs[2], **slider_props
        )
        
        # Visualization mode selection
        self.radio_mode = RadioButtons(
            plt.axes([0.05, 0.15, 0.15, 0.15]),
            ('3D Surface', 'Time Structure', 'Volatility', 'Local Vol', 'Density'),
            active=0
        )
        
        # Derivative mode toggle
        self.button_derivatives = Button(
            plt.axes([0.05, 0.08, 0.15, 0.04]), 
            'Analytical: ON',
            color='lightgreen'
        )
        
        # Benchmark button
        self.button_benchmark = Button(
            plt.axes([0.05, 0.03, 0.15, 0.04]), 
            'Run Benchmark',
            color='lightyellow'
        )
        
        # Connect callbacks
        self.connect_callbacks()
        
    def connect_callbacks(self):
        """Connect slider and button callbacks."""
        sliders = [
            self.slider_rho, self.slider_theta_inf, self.slider_theta_0,
            self.slider_kappa, self.slider_p0, self.slider_p1, self.slider_p2,
            self.slider_q1, self.slider_q2
        ]
        
        for slider in sliders:
            slider.on_changed(self.update_plot)
            
        self.radio_mode.on_clicked(self.change_mode)
        self.button_derivatives.on_clicked(self.toggle_derivatives)
        self.button_benchmark.on_clicked(self.run_benchmark)
        
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
        
    def validate_current_parameters(self) -> Tuple[bool, List[str]]:
        """Validate current parameter values."""
        return validate_parametric_ssvi_parameters(
            self.rho, self.theta_inf, self.theta_0, self.kappa,
            self.p_coeffs, self.q_coeffs
        )
        
    def update_plot(self, val=None):
        """Update the main plot based on current parameters and mode."""
        self.update_parameters()
        
        # Validate parameters
        is_valid, violations = self.validate_current_parameters()
        
        if not is_valid:
            self.show_parameter_violations(violations)
            return
            
        try:
            if self.current_mode == "3D Surface":
                self.plot_3d_surface()
            elif self.current_mode == "Time Structure":
                self.plot_time_structure()
            elif self.current_mode == "Volatility":
                self.plot_volatility_smile()
            elif self.current_mode == "Local Vol":
                self.plot_local_volatility()
            elif self.current_mode == "Density":
                self.plot_density()
                
            self.show_parameter_info()
            
        except Exception as e:
            self.show_error(f"Error in computation: {str(e)}")
            
        self.fig.canvas.draw()
        
    def plot_3d_surface(self):
        """Plot 3D total variance surface."""
        mu_values = np.linspace(-self.mu_range, self.mu_range, self.n_mu)
        T_values = np.linspace(self.T_min, self.T_max, self.n_T)
        
        MU, T_GRID = np.meshgrid(mu_values, T_values)
        W = np.zeros_like(MU)
        
        start_time = time.time()
        
        for i, T in enumerate(T_values):
            if self.analytical_mode:
                w, _, _, _ = compute_parametric_ssvi_all_derivatives(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
            else:
                w = compute_parametric_ssvi_total_variance(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
            W[i, :] = w
            
        computation_time = time.time() - start_time
        
        self.ax_main.clear()
        
        if hasattr(self.ax_main, 'zaxis'):
            self.ax_main.remove()
            self.ax_main = self.fig.add_subplot(122, projection='3d')
            
        from mpl_toolkits.mplot3d import Axes3D
        self.ax_main.remove()
        self.ax_main = self.fig.add_subplot(122, projection='3d')
        
        surf = self.ax_main.plot_surface(
            MU, T_GRID, W, cmap='viridis', alpha=0.8,
            linewidth=0, antialiased=False
        )
        
        self.ax_main.set_xlabel('Log-moneyness μ')
        self.ax_main.set_ylabel('Time to maturity T')
        self.ax_main.set_zlabel('Total variance w(μ,T)')
        self.ax_main.set_title(f'Parametric SSVI Surface\n(Computed in {computation_time:.3f}s)')
        
        self.fig.colorbar(surf, ax=self.ax_main, shrink=0.5)
        
    def plot_time_structure(self):
        """Plot term structure of variance."""
        self.ax_main.clear()
        
        T_values = np.linspace(self.T_min, self.T_max, 100)
        mu_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(mu_levels)))
        
        start_time = time.time()
        
        for mu, color in zip(mu_levels, colors):
            w_values = []
            for T in T_values:
                if self.analytical_mode:
                    w, _, _, _ = compute_parametric_ssvi_all_derivatives(
                        np.array([mu]), T, self.rho, self.theta_inf, self.theta_0,
                        self.kappa, self.p_coeffs, self.q_coeffs
                    )
                else:
                    w = compute_parametric_ssvi_total_variance(
                        np.array([mu]), T, self.rho, self.theta_inf, self.theta_0,
                        self.kappa, self.p_coeffs, self.q_coeffs
                    )
                w_values.append(w[0])
                
            self.ax_main.plot(T_values, w_values, color=color, 
                             label=f'μ = {mu:.1f}', linewidth=2)
            
        computation_time = time.time() - start_time
        
        self.ax_main.set_xlabel('Time to maturity T')
        self.ax_main.set_ylabel('Total variance w(μ,T)')
        self.ax_main.set_title(f'Term Structure of Variance\n(Computed in {computation_time:.3f}s)')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        
    def plot_volatility_smile(self):
        """Plot volatility smile for different maturities."""
        self.ax_main.clear()
        
        mu_values = np.linspace(-self.mu_range, self.mu_range, 100)
        T_levels = [0.25, 0.5, 1.0, 2.0]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(T_levels)))
        
        start_time = time.time()
        
        for T, color in zip(T_levels, colors):
            if self.analytical_mode:
                w, _, _, _ = compute_parametric_ssvi_all_derivatives(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
            else:
                w = compute_parametric_ssvi_total_variance(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
            
            # Convert to implied volatility
            sigma = np.sqrt(w / T)
            
            self.ax_main.plot(mu_values, sigma, color=color, 
                             label=f'T = {T:.2f}', linewidth=2)
            
        computation_time = time.time() - start_time
        
        self.ax_main.set_xlabel('Log-moneyness μ')
        self.ax_main.set_ylabel('Implied volatility σ(μ,T)')
        self.ax_main.set_title(f'Volatility Smile\n(Computed in {computation_time:.3f}s)')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        
    def plot_local_volatility(self):
        """Plot local volatility surface."""
        if not HAS_LOCAL_VOLATILITY:
            self.ax_main.clear()
            self.ax_main.text(0.5, 0.5, 'Local volatility module\nnot available', 
                             ha='center', va='center', transform=self.ax_main.transAxes)
            return
            
        self.ax_main.clear()
        
        mu_values = np.linspace(-1.5, 1.5, 30)
        T_values = np.linspace(0.1, 2.0, 20)
        
        MU, T_GRID = np.meshgrid(mu_values, T_values)
        LOCAL_VOL = np.zeros_like(MU)
        
        start_time = time.time()
        
        for i, T in enumerate(T_values):
            try:
                local_vol, is_valid, _ = compute_parametric_ssvi_local_volatility(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs, self.r
                )
                
                # Only use valid values
                LOCAL_VOL[i, :] = np.where(is_valid, local_vol, np.nan)
                
            except Exception as e:
                LOCAL_VOL[i, :] = np.nan
                
        computation_time = time.time() - start_time
        
        # Remove 3D projection if it exists
        if hasattr(self.ax_main, 'zaxis'):
            self.ax_main.remove()
            self.ax_main = self.fig.add_subplot(122)
            
        # Plot as contour
        valid_mask = ~np.isnan(LOCAL_VOL)
        if np.any(valid_mask):
            cs = self.ax_main.contourf(MU, T_GRID, LOCAL_VOL, levels=20, cmap='viridis')
            self.fig.colorbar(cs, ax=self.ax_main)
            
        self.ax_main.set_xlabel('Log-moneyness μ')
        self.ax_main.set_ylabel('Time to maturity T')
        self.ax_main.set_title(f'Local Volatility\n(Computed in {computation_time:.3f}s)')
        
    def plot_density(self):
        """Plot risk-neutral density for different maturities."""
        self.ax_main.clear()
        
        mu_values = np.linspace(-2.0, 2.0, 200)
        T_levels = [0.25, 0.5, 1.0, 2.0]
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(T_levels)))
        
        start_time = time.time()
        
        for T, color in zip(T_levels, colors):
            if self.analytical_mode:
                _, w_prime, w_double_prime, _ = compute_parametric_ssvi_all_derivatives(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
            else:
                _, w_prime, w_double_prime = compute_parametric_ssvi_derivatives(
                    mu_values, T, self.rho, self.theta_inf, self.theta_0,
                    self.kappa, self.p_coeffs, self.q_coeffs
                )
            
            # Compute density using formula: g(k) = (1 - k*w'/2 - w'²/4*(1/4 + 1/w'') + w''/2) / sqrt(2π*w)
            # This is a simplified approximation
            w = compute_parametric_ssvi_total_variance(
                mu_values, T, self.rho, self.theta_inf, self.theta_0,
                self.kappa, self.p_coeffs, self.q_coeffs
            )
            
            # Risk-neutral density approximation
            density = np.exp(-mu_values**2 / (2 * w)) / np.sqrt(2 * np.pi * w)
            density *= (1 - mu_values * w_prime / 2 + w_double_prime / 2)
            
            # Remove negative or invalid values
            density = np.maximum(density, 0)
            
            self.ax_main.plot(mu_values, density, color=color, 
                             label=f'T = {T:.2f}', linewidth=2)
            
        computation_time = time.time() - start_time
        
        self.ax_main.set_xlabel('Log-moneyness μ')
        self.ax_main.set_ylabel('Risk-neutral density')
        self.ax_main.set_title(f'Risk-Neutral Density\n(Computed in {computation_time:.3f}s)')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        
    def change_mode(self, label):
        """Change visualization mode."""
        self.current_mode = label
        self.update_plot()
        
    def toggle_derivatives(self, event):
        """Toggle between analytical and numerical derivatives."""
        self.analytical_mode = not self.analytical_mode
        
        if self.analytical_mode:
            self.button_derivatives.label.set_text('Analytical: ON')
            self.button_derivatives.color = 'lightgreen'
        else:
            self.button_derivatives.label.set_text('Analytical: OFF')
            self.button_derivatives.color = 'lightcoral'
            
        self.update_plot()
        
    def run_benchmark(self, event):
        """Run performance benchmark comparing analytical vs numerical derivatives."""
        self.ax_info.clear()
        self.ax_info.text(0.5, 0.5, 'Running benchmark...', 
                         ha='center', va='center', transform=self.ax_info.transAxes)
        self.fig.canvas.draw()
        
        # Test parameters
        mu_values = np.linspace(-2.0, 2.0, 100)
        T_values = [0.25, 0.5, 1.0, 2.0]
        n_runs = 10
        
        results = {}
        
        for method in ['analytical', 'numerical']:
            times = []
            
            for _ in range(n_runs):
                start_time = time.time()
                
                for T in T_values:
                    if method == 'analytical':
                        compute_parametric_ssvi_all_derivatives(
                            mu_values, T, self.rho, self.theta_inf, self.theta_0,
                            self.kappa, self.p_coeffs, self.q_coeffs
                        )
                    else:
                        compute_parametric_ssvi_derivatives(
                            mu_values, T, self.rho, self.theta_inf, self.theta_0,
                            self.kappa, self.p_coeffs, self.q_coeffs
                        )
                        
                times.append(time.time() - start_time)
                
            results[method] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'times': times
            }
            
        # Store benchmark data
        self.benchmark_data = results
        
        # Display results
        self.show_benchmark_results()
        
    def show_benchmark_results(self):
        """Display benchmark results in the info area."""
        self.ax_info.clear()
        
        analytical = self.benchmark_data['analytical']
        numerical = self.benchmark_data['numerical']
        
        speedup = numerical['mean_time'] / analytical['mean_time']
        
        info_text = f"""Performance Benchmark Results:
        
Analytical Derivatives: {analytical['mean_time']:.4f} ± {analytical['std_time']:.4f} s
Numerical Derivatives:  {numerical['mean_time']:.4f} ± {numerical['std_time']:.4f} s

Speedup: {speedup:.2f}x faster with analytical derivatives
        
Current Mode: {'Analytical' if self.analytical_mode else 'Numerical'}
"""
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=9)
        
    def show_parameter_info(self):
        """Display current parameter information."""
        if hasattr(self, 'benchmark_data') and self.benchmark_data:
            self.show_benchmark_results()
            return
            
        # Analyze current parameters
        analysis = analyze_parametric_ssvi_properties(
            self.rho, self.theta_inf, self.theta_0, self.kappa,
            self.p_coeffs, self.q_coeffs
        )
        
        info_text = f"""Current Parameters:
        
ρ = {self.rho:.3f}, θ∞ = {self.theta_inf:.3f}, θ0 = {self.theta_0:.3f}, κ = {self.kappa:.3f}
p = [{self.p_coeffs[0]:.2f}, {self.p_coeffs[1]:.2f}, {self.p_coeffs[2]:.2f}]
q = [1.00, {self.q_coeffs[1]:.2f}, {self.q_coeffs[2]:.2f}]

Properties:
θ(0.25) = {analysis.get('theta_quarter', 'N/A'):.4f}
θ(1.0) = {analysis.get('theta_one', 'N/A'):.4f}
φ(θ∞) = {analysis.get('phi_theta_inf', 'N/A'):.4f}

Mode: {'Analytical' if self.analytical_mode else 'Numerical'} derivatives
"""
        
        self.ax_info.clear()
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=9)
        
    def show_parameter_violations(self, violations: List[str]):
        """Show parameter validation errors."""
        self.ax_main.clear()
        
        error_text = "Parameter Violations:\n\n" + "\n".join(f"• {v}" for v in violations)
        
        self.ax_main.text(0.5, 0.5, error_text, ha='center', va='center',
                         transform=self.ax_main.transAxes, color='red',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
    def show_error(self, error_msg: str):
        """Show computation error."""
        self.ax_main.clear()
        self.ax_main.text(0.5, 0.5, f"Error:\n{error_msg}", ha='center', va='center',
                         transform=self.ax_main.transAxes, color='red')
        
    def run(self):
        """Run the interactive application."""
        # Initial plot
        self.update_plot()
        
        # Show the plot
        plt.show()


def main():
    """Main function to run the enhanced parametric SSVI application."""
    print("Enhanced Parametric SSVI Interactive Application")
    print("=" * 50)
    print("Features:")
    print("- 5 visualization modes: 3D Surface, Time Structure, Volatility, Local Vol, Density")
    print("- Analytical vs numerical derivative comparison")
    print("- Performance benchmarking")
    print("- Real-time parameter validation")
    print("- Enhanced diagnostics")
    print()
    print("Controls:")
    print("- Use sliders to adjust parameters")
    print("- Select visualization mode with radio buttons")
    print("- Toggle derivative computation method")
    print("- Run performance benchmarks")
    print()
    print("Starting application...")
    
    app = EnhancedParametricSSVIApp()
    app.run()


if __name__ == "__main__":
    main()
