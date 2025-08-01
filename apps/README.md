# Interactive Applications

This directory contains GUI applications and interactive tools for visualizing and exploring SVI and parametric SSVI volatility surfaces.

## Applications

### `parametric_ssvi_iv_lv_density_app.py` ⭐ **Main Application**
**Purpose**: Complete interactive visualization of parametric SSVI with IV, LV, and density plots.

**Features**:
- Real-time parameter adjustment with sliders
- Implied volatility line plots with consistent viridis colormap
- Local volatility line plots (improved from contour surface)
- Risk-neutral density surface plots
- Parameter validation and arbitrage-free surface checking
- Centered two-column slider layout

**Usage**: 
```bash
cd apps
python parametric_ssvi_iv_lv_density_app.py
```

### `parametric_ssvi_app.py`
**Purpose**: Basic parametric SSVI application with simpler interface.

**Features**:
- Core parametric SSVI functionality
- Basic parameter controls
- Essential visualizations

### `enhanced_parametric_ssvi_app.py`
**Purpose**: Extended version with additional features and analysis tools.

**Features**:
- Enhanced parameter controls
- Additional validation metrics
- Extended analysis capabilities

### `simple_parametric_ssvi_app.py`
**Purpose**: Simplified version for educational purposes and quick testing.

**Features**:
- Minimal interface
- Essential SSVI functionality
- Easy to understand and modify

### `volatility_surface_app.py`
**Purpose**: General volatility surface visualization tool.

**Features**:
- Multiple volatility model support
- Surface plotting capabilities
- Cross-model comparisons

## Running Applications

### Prerequisites
Make sure you have the required dependencies installed:
```bash
pip install numpy matplotlib scipy
```

### Individual Applications
```bash
# Main interactive application (recommended)
cd apps
python parametric_ssvi_iv_lv_density_app.py

# Basic application
python parametric_ssvi_app.py

# Other applications
python enhanced_parametric_ssvi_app.py
python simple_parametric_ssvi_app.py
python volatility_surface_app.py
```

### From Root Directory
You can also run applications from the root directory:
```bash
python -m apps.parametric_ssvi_iv_lv_density_app
python -m apps.parametric_ssvi_app
```

## Application Features

### Interactive Controls
- **Parameter Sliders**: Real-time adjustment of all model parameters
- **Validation**: Automatic parameter constraint checking
- **Visual Feedback**: Immediate plot updates with parameter changes

### Visualization Options
- **Implied Volatility**: Line plots with time-to-maturity color coding
- **Local Volatility**: Dupire local volatility computation and display
- **Risk-Neutral Density**: Surface plots with mathematical validation

### Mathematical Validation
- **Arbitrage-Free Checking**: Validates positive total variance and second derivatives
- **Parameter Constraints**: Enforces mathematical bounds (ρ ∈ (-1,1), etc.)
- **Numerical Stability**: Handles edge cases and provides warnings

## Tips for Usage

1. **Start with Default Parameters**: The applications load with reasonable default values
2. **Parameter Exploration**: Use sliders to understand parameter sensitivity
3. **Arbitrage Checking**: Watch for validation messages to ensure valid surfaces
4. **Performance**: The main application uses analytical derivatives for speed
5. **Saving Results**: Applications support saving plots and parameter configurations

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the correct directory and that the parent modules are accessible.

### Performance Issues
- The main application uses analytical derivatives for optimal performance
- For very fine grids, consider reducing the number of points
- Close unused plot windows to free memory

### Mathematical Warnings
- Parameter constraint violations will show warnings
- Negative densities indicate potential arbitrage opportunities
- Small denominators in φ(θ) are handled with warnings
