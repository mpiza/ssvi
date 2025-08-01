import numpy as np
import matplotlib

# Use a non-interactive backend when this script is imported.  If run directly,
# the default interactive backend will be used and interactivity will work as
# expected in a Jupyter notebook or a Python session with GUI support.  This
# call has no effect if a GUI backend is already active.
matplotlib.use(matplotlib.get_backend())
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Import the separated core calculation modules
from svi_models import compute_svi_surface, compute_ssvi_surface, compute_svi_volatility_smile
from density_analysis import compute_svi_risk_neutral_density, verify_density_properties


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
    vol = compute_svi_surface(k_vals, t_vals, **init_params)

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
        new_vol = compute_svi_surface(k_vals, t_vals, **params)
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
    vol = compute_ssvi_surface(k_vals, t_vals, **init_params)

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
        new_vol = compute_ssvi_surface(k_vals, t_vals, **params)
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

    vol_smile = compute_svi_volatility_smile(k_vals, **init_params)
    density = compute_svi_risk_neutral_density(k_vals, **init_params)
    
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
        new_vol_smile = compute_svi_volatility_smile(k_vals, **params)
        new_density = compute_svi_risk_neutral_density(k_vals, **params)
        
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