# Analysis Tools

This directory contains performance analysis, comparison tools, and benchmarking scripts for SVI models.

## Analysis Scripts

### `performance_analysis.py`
**Purpose**: Benchmarking and performance analysis of different SVI implementations.

**Features**:
- Speed comparison between analytical and numerical derivatives
- Memory usage analysis
- Accuracy assessment
- Performance profiling across different parameter ranges

**Usage**:
```bash
cd analysis
python performance_analysis.py
```

**Output**:
- Performance comparison plots
- Timing benchmarks
- Accuracy metrics
- Memory usage reports

### `quick_iv_lv_comparison.py`
**Purpose**: Quick comparison tool for implied vs local volatility analysis.

**Features**:
- Side-by-side IV and LV plots
- Parameter sensitivity analysis
- Quick validation of Dupire formula implementation
- Cross-model comparisons

**Usage**:
```bash
cd analysis
python quick_iv_lv_comparison.py
```

### `test_derivatives.py`
**Purpose**: Comprehensive testing and validation of derivative implementations.

**Features**:
- Analytical vs numerical derivative comparison
- Accuracy validation across parameter space
- Edge case testing
- Mathematical property verification

**Usage**:
```bash
cd analysis
python test_derivatives.py
```

## Analysis Types

### Performance Analysis
- **Speed Benchmarks**: Compare execution times between methods
- **Memory Profiling**: Analyze memory usage patterns
- **Scalability**: Test performance with varying grid sizes
- **Optimization**: Identify bottlenecks and optimization opportunities

### Accuracy Analysis
- **Numerical Precision**: Validate analytical formulas against numerical methods
- **Error Propagation**: Analyze how errors propagate through calculations
- **Stability Testing**: Test numerical stability under extreme conditions
- **Convergence Analysis**: Study convergence properties of numerical methods

### Comparative Analysis
- **Model Comparison**: Compare different SVI implementations
- **Method Comparison**: Analytical vs numerical approaches
- **Parameter Sensitivity**: How sensitive models are to parameter changes
- **Cross-Validation**: Validate results across different implementations

## Running Analysis

### Individual Scripts
```bash
cd analysis

# Performance benchmarking
python performance_analysis.py

# Quick IV/LV comparison
python quick_iv_lv_comparison.py

# Derivative testing
python test_derivatives.py
```

### Batch Analysis
```bash
cd analysis
for script in *.py; do
    echo "Running $script..."
    python "$script"
done
```

## Output Files

### Performance Analysis
- `performance_comparison.png`: Speed and accuracy comparisons
- `memory_usage.png`: Memory consumption analysis
- `timing_results.csv`: Detailed timing data

### Comparison Analysis
- `iv_lv_comparison.png`: Implied vs local volatility plots
- `parameter_sensitivity.png`: Sensitivity analysis results
- `cross_validation.png`: Cross-validation results

### Derivative Testing
- `derivative_accuracy.png`: Accuracy validation plots
- `numerical_stability.png`: Stability test results
- `error_analysis.csv`: Detailed error metrics

## Interpretation Guidelines

### Performance Metrics
- **Execution Time**: Lower is better, measured in seconds or milliseconds
- **Memory Usage**: Lower is better, measured in MB
- **Accuracy**: Higher is better, measured in relative error
- **Stability**: Consistent results across parameter ranges

### Analysis Results
- **Green Indicators**: Good performance/accuracy
- **Yellow Indicators**: Acceptable but could be improved
- **Red Indicators**: Issues that need attention

### Recommendations
Based on analysis results, the scripts provide recommendations for:
- Optimal parameter ranges
- Best implementation choices
- Performance optimization opportunities
- Areas needing improvement

## Advanced Usage

### Custom Analysis
Modify scripts to analyze specific scenarios:
```python
# Example: Custom parameter ranges
param_ranges = {
    'rho': np.linspace(-0.9, 0.9, 20),
    'theta_inf': np.linspace(0.1, 1.0, 10),
    # ... other parameters
}
```

### Batch Testing
Run analysis across multiple parameter sets:
```python
# Example: Parameter sweep
for param_set in parameter_combinations:
    results = run_analysis(param_set)
    save_results(results, param_set)
```

### Performance Profiling
Use Python profiling tools with analysis scripts:
```bash
python -m cProfile -o profile_output.prof performance_analysis.py
python -m pstats profile_output.prof
```

## Tips for Analysis

1. **Baseline Comparison**: Always compare against known good results
2. **Parameter Validation**: Ensure parameters are within valid ranges
3. **Statistical Significance**: Run multiple iterations for reliable statistics
4. **Documentation**: Document analysis parameters and conditions
5. **Reproducibility**: Use fixed random seeds for reproducible results
