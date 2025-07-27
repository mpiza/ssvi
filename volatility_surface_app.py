import numpy as np
import matplotlib

# Use a non-interactive backend when this script is imported.  If run directly,
# the default interactive backend will be used and interactivity will work as
# expected in a Jupyter notebook or a Python session with GUI support.  This
# call has no effect if a GUI backend is already active.
matplotlib.use(matplotlib.get_backend())
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def _compute_svi_surface(k_values, t_values, a, b, rho, m, sigma):
    """Compute the SVI volatility surface.

    Parameters
    ----------
    k_values : ndarray
        1-D array of log‑moneyness values.
    t_values : ndarray
        1-D array of time to maturity values.
    a, b, rho, m, sigma : float
        SVI parameters.  See Gatheral (2004) for details.

    Returns
    -------
    vol_surface : ndarray
        2-D array with shape (len(t_values), len(k_values)) containing
        Black implied volatilities computed from the SVI total variance
        parameterisation.
    """
    # total variance as a function of k only
    diff = k_values - m
    total_variance_k = a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))

    # Expand across maturities: each maturity sees the same total variance function
    # but implied volatility scales by sqrt(w / T)
    # Avoid division by zero by ensuring T > 0
    T_grid, K_grid = np.meshgrid(t_values, k_values, indexing="ij")
    W_grid = total_variance_k[np.newaxis, :].repeat(len(t_values), axis=0)
    vol_surface = np.sqrt(np.maximum(W_grid, 0.0) / T_grid)
    return vol_surface


def _compute_ssvi_surface(k_values, t_values, theta, phi, rho):
    """Compute the SSVI volatility surface.

    The SSVI (Surface SVI) total variance is defined as

        w(k) = 0.5 * theta * (1 + rho * phi * k + sqrt((phi * k + rho)^2 + 1 - rho^2)),

    where theta > 0, phi > 0, and -1 < rho < 1.  The implied volatility is
    derived by sigma(k, T) = sqrt(w(k) / T) for maturity T.

    Parameters
    ----------
    k_values : ndarray
        1-D array of log‑moneyness values.
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
    # total variance as a function of k only
    kk = phi * k_values
    inside = (kk + rho) ** 2 + 1.0 - rho ** 2
    # ensure numerical stability by clipping the radicand at zero
    rad = np.sqrt(np.maximum(inside, 0.0))
    w_k = 0.5 * theta * (1.0 + rho * kk + rad)

    # Build the grid and compute implied vols as with SVI
    T_grid, K_grid = np.meshgrid(t_values, k_values, indexing="ij")
    W_grid = w_k[np.newaxis, :].repeat(len(t_values), axis=0)
    vol_surface = np.sqrt(np.maximum(W_grid, 0.0) / T_grid)
    return vol_surface


def run_svi_interactive():
    """Launch an interactive 3D plot for the SVI volatility surface.

    This function opens a Matplotlib window containing a 3D surface plot of
    Black implied volatilities generated from the SVI parameterisation.  It
    provides sliders for each of the five SVI parameters (a, b, rho, m, sigma)
    as well as time to maturity, allowing the user to adjust parameters and
    immediately see their effect on the surface.
    """
    # Define the grid for log‑moneyness and maturities
    k_vals = np.linspace(-2.0, 2.0, 50)
    t_vals = np.linspace(0.1, 2.0, 50)

    # Set reasonable starting values for the SVI parameters
    init_params = {
        "a": 0.02,
        "b": 0.4,
        "rho": -0.2,
        "m": 0.0,
        "sigma": 0.4,
    }

    # Compute the initial surface
    vol = _compute_svi_surface(k_vals, t_vals, **init_params)

    # Create the figure and axes
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the initial surface
    K_grid, T_grid = np.meshgrid(k_vals, t_vals)
    surf = ax.plot_surface(K_grid, T_grid, vol, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_xlabel("log‑moneyness (k)")
    ax.set_ylabel("maturity (T, years)")
    ax.set_zlabel("implied vol")
    ax.set_title("SVI volatility surface")

    # Adjust layout to make room for sliders
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Create axes for sliders
    slider_axs = {}
    param_ranges = {
        "a": (0.0, 0.5),  # intercept
        "b": (0.0, 1.0),  # slope parameter
        "rho": (-0.99, 0.99),  # correlation parameter
        "m": (-1.0, 1.0),  # horizontal shift
        "sigma": (0.0, 1.0),  # curvature parameter
    }

    # Place sliders vertically stacked below the plot
    slider_y_start = 0.15
    slider_height = 0.03
    slider_spacing = 0.05
    for i, (param, (p_min, p_max)) in enumerate(param_ranges.items()):
        ax_slider = plt.axes([0.15, slider_y_start - i * slider_spacing, 0.65, slider_height], facecolor="lightgoldenrodyellow")
        slider = Slider(
            ax=ax_slider,
            label=f"{param}",
            valmin=p_min,
            valmax=p_max,
            valinit=init_params[param],
        )
        slider_axs[param] = slider

    def update(val=None):
        # Read current slider values
        params = {p: slider_axs[p].val for p in param_ranges.keys()}
        # Recompute the surface
        new_vol = _compute_svi_surface(k_vals, t_vals, **params)
        # Clear and replot surface
        ax.clear()
        surf = ax.plot_surface(K_grid, T_grid, new_vol, cmap="viridis", linewidth=0, antialiased=False)
        ax.set_xlabel("log‑moneyness (k)")
        ax.set_ylabel("maturity (T, years)")
        ax.set_zlabel("implied vol")
        ax.set_title("SVI volatility surface")
        fig.canvas.draw_idle()

    # Attach update callback to sliders
    for slider in slider_axs.values():
        slider.on_changed(update)

    # Add a reset button to restore default parameter values
    reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, "Reset", hovercolor="0.975")

    def reset(event):
        for param, default in init_params.items():
            slider_axs[param].reset()
        update()
    reset_button.on_clicked(reset)

    plt.show()


def run_ssvi_interactive():
    """Launch an interactive 3D plot for the SSVI volatility surface.

    This function opens a Matplotlib window containing a 3D surface plot of
    Black implied volatilities generated from the SSVI parameterisation.  It
    provides sliders for the three SSVI parameters (theta, phi, rho) and
    maturity, allowing the user to adjust parameters and observe their
    influence on the surface in real time.
    """
    # Define the grid for log‑moneyness and maturities
    k_vals = np.linspace(-2.0, 2.0, 50)
    t_vals = np.linspace(0.1, 2.0, 50)

    # Initial parameter values for SSVI
    init_params = {
        "theta": 0.3,
        "phi": 0.5,
        "rho": -0.2,
    }

    # Compute the initial surface
    vol = _compute_ssvi_surface(k_vals, t_vals, **init_params)

    # Create the figure and axes
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the initial surface
    K_grid, T_grid = np.meshgrid(k_vals, t_vals)
    surf = ax.plot_surface(K_grid, T_grid, vol, cmap="plasma", linewidth=0, antialiased=False)
    ax.set_xlabel("log‑moneyness (k)")
    ax.set_ylabel("maturity (T, years)")
    ax.set_zlabel("implied vol")
    ax.set_title("SSVI volatility surface")

    # Adjust layout to make room for sliders
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Define ranges for SSVI parameters
    param_ranges = {
        "theta": (0.01, 1.0),
        "phi": (0.1, 2.0),
        "rho": (-0.99, 0.99),
    }

    slider_axs = {}
    slider_y_start = 0.15
    slider_height = 0.03
    slider_spacing = 0.05
    for i, (param, (p_min, p_max)) in enumerate(param_ranges.items()):
        ax_slider = plt.axes([0.15, slider_y_start - i * slider_spacing, 0.65, slider_height], facecolor="lightgoldenrodyellow")
        slider = Slider(
            ax=ax_slider,
            label=f"{param}",
            valmin=p_min,
            valmax=p_max,
            valinit=init_params[param],
        )
        slider_axs[param] = slider

    def update(val=None):
        # Gather current slider values
        params = {p: slider_axs[p].val for p in param_ranges.keys()}
        # Recompute the surface
        new_vol = _compute_ssvi_surface(k_vals, t_vals, **params)
        # Clear and draw updated surface
        ax.clear()
        surf = ax.plot_surface(K_grid, T_grid, new_vol, cmap="plasma", linewidth=0, antialiased=False)
        ax.set_xlabel("log‑moneyness (k)")
        ax.set_ylabel("maturity (T, years)")
        ax.set_zlabel("implied vol")
        ax.set_title("SSVI volatility surface")
        fig.canvas.draw_idle()

    # Attach callback
    for slider in slider_axs.values():
        slider.on_changed(update)

    # Add a reset button
    reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, "Reset", hovercolor="0.975")

    def reset(event):
        for param, default in init_params.items():
            slider_axs[param].reset()
        update()
    reset_button.on_clicked(reset)

    plt.show()


def run_svi_smile_interactive():
    """Launch an interactive 2D plot for the SVI volatility smile at a single maturity.

    This function opens a Matplotlib window containing two subplots:
    1. Top: SVI volatility smile as a function of log-moneyness
    2. Bottom: Risk-neutral density function derived from the smile
    
    It provides sliders for each of the five SVI parameters (a, b, rho, m, sigma)
    and the maturity, allowing the user to see how each parameter affects the
    volatility smile shape and implied probability distribution.
    """
    # Define the grid for log‑moneyness - use reasonable range focused on typical smile behavior
    k_vals = np.linspace(-3.0, 3.0, 300)
    
    # Set reasonable starting values for the SVI parameters and maturity
    init_params = {
        "a": 0.02,
        "b": 0.4,
        "rho": -0.2,
        "m": 0.0,
        "sigma": 0.4,
        "T": 0.5,  # maturity in years
    }

    # Compute the initial volatility smile
    def compute_svi_smile(k_values, T, a, b, rho, m, sigma):
        """Compute SVI volatility smile for a single maturity."""
        diff = k_values - m
        total_variance = a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))
        # Ensure non-negative total variance and compute implied vol
        vol_smile = np.sqrt(np.maximum(total_variance, 0.0) / T)
        return vol_smile

    def compute_risk_neutral_density(k_values, T, a, b, rho, m, sigma):
        """Compute risk-neutral density from SVI total variance using correct Breeden-Litzenberger formula.
        
        This implements the complete derivation with the shape function g(y) that accounts for
        the curvature and derivatives of the SVI total variance function.
        
        Following the derivation:
        w(y) = a + b[ρ(y-m) + √((y-m)² + σ²)]
        w'(y) = b[ρ + (y-m)/√((y-m)² + σ²)]
        w''(y) = b σ² / ((y-m)² + σ²)^(3/2)
        
        g(y) = [1 - y w'(y)/(2w(y))]² - (w'(y))²/4 [1/w(y) + 1/4] + w''(y)/2
        
        p(y) = g(y)/√(2πw(y)) * exp(-d₋²/2)
        where d₋ = -y/√w(y) - √w(y)/2
        
        f_S(x) = (1/x) * p(y) where y = ln(x/F)
        
        For log-moneyness density: ρ(k) = x * f_S(x) = F * exp(k) * f_S(F * exp(k)) = p(k)
        """
        # SVI total variance and its derivatives
        y = k_values  # log-moneyness (using k instead of y for consistency with code)
        diff = y - m
        sqrt_term = np.sqrt(diff**2 + sigma**2)
        
        # Total variance w(y)
        w = a + b * (rho * diff + sqrt_term)
        
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



    def verify_density_properties(k_values, density):
        """Verify that the computed density has proper mathematical properties."""
        dk = k_values[1] - k_values[0]
        
        # Test 1: Check for negative densities (ARBITRAGE INDICATOR!)
        negative_mask = density < 0
        if np.any(negative_mask):
            num_negative = np.sum(negative_mask)
            min_density = np.min(density)
            negative_k_range = [np.min(k_values[negative_mask]), np.max(k_values[negative_mask])]
            print(f"🚨 ARBITRAGE DETECTED! 🚨")
            print(f"Found {num_negative} points with NEGATIVE density!")
            print(f"Minimum density: {min_density:.6f}")
            print(f"Negative density occurs in k-range: [{negative_k_range[0]:.2f}, {negative_k_range[1]:.2f}]")
        else:
            print(f"✓ All density values are non-negative")
        
        # Test 2: Probability conservation (should integrate to 1)
        # Only integrate positive part for meaningful probability
        positive_density = np.maximum(density, 0.0)
        total_prob = np.trapz(positive_density, k_values)
        print(f"Total probability (positive part): {total_prob:.6f}")
        
        # Test 3: Full integral (including negative parts)
        full_integral = np.trapz(density, k_values)
        print(f"Full integral (including negatives): {full_integral:.6f}")
        
        # Test 4: Finite values
        has_infinite = np.any(np.isinf(density))
        has_nan = np.any(np.isnan(density))
        print(f"Contains infinite values: {has_infinite}")
        print(f"Contains NaN values: {has_nan}")
        
        # Test 5: Expected value (first moment, only positive part)
        if total_prob > 0:
            expected_k = np.trapz(k_values * positive_density, k_values) / total_prob
            print(f"Expected log-moneyness (positive part): {expected_k:.6f}")
        
        return total_prob, np.min(density), not (has_infinite or has_nan)

    vol_smile = compute_svi_smile(k_vals, **init_params)
    density = compute_risk_neutral_density(k_vals, **init_params)
    
    # Verify the initial density
    print("=== Density Verification ===")
    verify_density_properties(k_vals, density)
    print("============================")

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot the initial smile (top subplot)
    line1, = ax1.plot(k_vals, vol_smile, 'b-', linewidth=2, label='SVI volatility smile')
    ax1.set_xlabel("log‑moneyness (k)")
    ax1.set_ylabel("implied volatility")
    ax1.set_title(f"SVI volatility smile (T = {init_params['T']:.2f} years)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot the initial density (bottom subplot)
    line2, = ax2.plot(k_vals, density, 'r-', linewidth=2, label='Risk-neutral density')
    ax2.set_xlabel("log‑moneyness (k)")
    ax2.set_ylabel("probability density")
    ax2.set_title(f"Risk-neutral density (T = {init_params['T']:.2f} years)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Adjust layout to make room for sliders
    plt.subplots_adjust(left=0.1, bottom=0.35, hspace=0.3)

    # Create axes for sliders
    slider_axs = {}
    param_ranges = {
        "a": (0.0, 0.5),      # intercept
        "b": (0.0, 1.0),      # slope parameter
        "rho": (-0.99, 0.99), # correlation parameter
        "m": (-1.0, 1.0),     # horizontal shift
        "sigma": (0.0, 1.0),  # curvature parameter
        "T": (0.1, 2.0),      # maturity
    }

    # Place sliders vertically stacked below the plot
    slider_y_start = 0.3
    slider_height = 0.03
    slider_spacing = 0.04
    for i, (param, (p_min, p_max)) in enumerate(param_ranges.items()):
        ax_slider = plt.axes([0.15, slider_y_start - i * slider_spacing, 0.65, slider_height], 
                           facecolor="lightgoldenrodyellow")
        slider = Slider(
            ax=ax_slider,
            label=f"{param}",
            valmin=p_min,
            valmax=p_max,
            valinit=init_params[param],
        )
        slider_axs[param] = slider

    def update(val=None):
        # Read current slider values
        params = {p: slider_axs[p].val for p in param_ranges.keys()}
        # Recompute the smile and density
        new_vol_smile = compute_svi_smile(k_vals, **params)
        new_density = compute_risk_neutral_density(k_vals, **params)
        
        # Update the smile line data (top subplot)
        line1.set_ydata(new_vol_smile)
        ax1.set_title(f"SVI volatility smile (T = {params['T']:.2f} years)")
        ax1.relim()
        ax1.autoscale_view(scalex=False, scaley=True)
        
        # Update the density line data (bottom subplot)
        line2.set_ydata(new_density)
        ax2.set_title(f"Risk-neutral density (T = {params['T']:.2f} years)")
        ax2.relim()
        ax2.autoscale_view(scalex=False, scaley=True)
        
        fig.canvas.draw_idle()

    # Attach update callback to sliders
    for slider in slider_axs.values():
        slider.on_changed(update)

    # Add a reset button to restore default parameter values
    reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, "Reset", hovercolor="0.975")

    def reset(event):
        for param, default in init_params.items():
            slider_axs[param].reset()
        update()
    reset_button.on_clicked(reset)

    plt.show()


if __name__ == "__main__":
    # Simple CLI interface to choose which surface to display
    print("Select the volatility surface to visualize:")
    print("  1) SVI surface (a, b, rho, m, sigma)")
    print("  2) SSVI surface (theta, phi, rho)")
    print("  3) SVI smile at single maturity (a, b, rho, m, sigma, T)")
    choice = input("Enter 1, 2, or 3: ").strip()
    if choice == "1":
        run_svi_interactive()
    elif choice == "2":
        run_ssvi_interactive()
    elif choice == "3":
        run_svi_smile_interactive()
    else:
        print("Invalid selection.")