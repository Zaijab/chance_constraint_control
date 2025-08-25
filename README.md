# Chance-Constrained Gaussian Mixture Steering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-enabled-green.svg)](https://github.com/google/jax)

This repository reproduces and extends the methodology from **"Chance-Constrained Gaussian Mixture Steering to a Terminal Gaussian Distribution"** by Kumagai & Oguri (2024 IEEE CDC).

## Overview

The implementation addresses finite-horizon optimal control of discrete-time linear systems with probabilistic constraints:

- **Initial State**: Gaussian Mixture Model (GMM) distribution capturing multi-modal uncertainty
- **Terminal Objective**: Steer to a specified target Gaussian distribution
- **Chance Constraints**: Probabilistic bounds on state and control trajectories
- **Solution Method**: Convex optimization via Iterative Risk Allocation (IRA)

### Mathematical Problem Formulation

Given a discrete-time linear system:
```
x_{k+1} = A_k x_k + B_k u_k + w_k
```

**Objective**: Find control policy that minimizes expected cost while satisfying:
- `P(H_k x_k ≤ h_k) ≥ 1 - Δ_k` (state chance constraints)
- `P(||u_k|| ≤ γ_k) ≥ 1 - Γ_k` (control chance constraints)  
- Terminal distribution constraint: `x_N ~ N(μ_f, Σ_f)`

## Key Features

- **Modern JAX Implementation**: Automatic differentiation, JIT compilation, and GPU acceleration
- **Equinox Integration**: Neural network-style modeling of control policies and system dynamics
- **Convex Optimization**: CVXPy integration for deterministic chance constraint reformulation
- **Complete Paper Reproduction**: All numerical examples from the original paper
- **Monte Carlo Validation**: Statistical verification of theoretical results
- **Extensible Architecture**: Support for additional dynamical systems and constraint types

## Installation

### Prerequisites
- Python 3.11 or higher
- CUDA support (optional, for GPU acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Zaijab/chance_constraint_control.git
cd chance_constraint_control

# Install with development dependencies
pip install -e ".[dev]"
```

### Dependencies
Core dependencies are automatically managed via `pyproject.toml`:
- **JAX ecosystem**: `jax`, `equinox`, `optax`, `jaxopt`
- **Optimization**: `cvxpy`, `scipy`
- **Scientific computing**: `numpy`, `matplotlib`, `tensorflow_probability`
- **Development tools**: `pytest`, `black`, `isort`, `mypy`

## Quick Start

### Reproduce Paper Results

```bash
python examples/reproduce_paper_results.py
```

This script reproduces the 4D double integrator example from the paper, comparing:
- Unconstrained solution
- Uniform Risk Allocation (URA) 
- Iterative Risk Allocation (IRA)

### Basic Usage

```python
import jax.numpy as jnp
from chance_control import (
    create_double_integrator_2d,
    create_paper_initial_gmm,
    create_paper_target_gaussian,
    ConvexOptimizationProblem,
    IterativeRiskAllocation
)

# Create 2D double integrator system (4D state: [x, y, vx, vy])
horizon = 20
system = create_double_integrator_2d(horizon, dt=0.2)

# Initial 3-component GMM distribution
initial_gmm = create_paper_initial_gmm()

# Target terminal Gaussian distribution
target_mean, target_cov = create_paper_target_gaussian()

# Solve with IRA
ira = IterativeRiskAllocation(
    system, initial_gmm, target_mean, target_cov, horizon
)

success, solution, info = ira.run_ira(
    constraint_matrices=jnp.array([[1.3, -1, 0, 0], [-1, 1, 0, 0]]),
    constraint_bounds=jnp.array([11.0, -1.0]),
    control_bound=6.5,
    cost_type="quadratic"
)

if success:
    print(f"Converged after {info['iterations']} iterations")
    print(f"Final cost: {info['final_cost']:.6f}")
```

## Architecture

### Core Components

- **`src/chance_control/linear_system.py`** - Discrete-time linear system implementation
- **`src/chance_control/gmm_models.py`** - Gaussian mixture model representation and propagation
- **`src/chance_control/control_policy.py`** - Affine feedback policy with probabilistic selection
- **`src/chance_control/chance_constraints.py`** - Deterministic convex reformulation of chance constraints
- **`src/chance_control/optimization.py`** - Convex optimization framework
- **`src/chance_control/risk_allocation.py`** - Iterative Risk Allocation algorithm

### Key Theoretical Results Implemented

1. **Proposition 3**: State distribution under proposed control policy
2. **Proposition 4**: Control distribution under affine policy
3. **Proposition 6**: Terminal constraint sufficient condition  
4. **Theorem 2**: Deterministic chance constraint reformulation
5. **IRA Algorithm**: Risk allocation refinement for improved optimality

## Paper Reproduction

The numerical example reproduces:

**System**: 4D double integrator (2D position + velocity)
```
x_{k+1} = [1  0  Δt  0 ] x_k + [Δt²/2   0  ] u_k
          [0  1  0   Δt]       [0     Δt²/2]
          [0  0  1   0 ]       [Δt      0  ]
          [0  0  0   1 ]       [0      Δt  ]
```

**Initial GMM**: 3 components with weights (0.3, 0.4, 0.3)
- μ₁ = [5, -1, 5, 0]ᵀ, μ₂ = [3.5, 0.5, 8, 0]ᵀ, μ₃ = [4, -0.5, 7, 0]ᵀ
- Σᵢ = diag(0.05, 0.05, 0.01, 0.01) for all i

**Target**: μf = [8, 5.5, 0, 0]ᵀ, Σf = diag(0.05, 0.05, 0.01, 0.01)

**Constraints**:
- State: 1.3x - y ≤ 11, -x + y ≤ -1 (Δ = 0.005)
- Control: ‖u‖ ≤ 6.5 (Γ = 0.005)

**Results**: ~5% cost improvement with IRA over URA

## Development

### Testing

Run the full test suite:
```bash
python -m pytest tests/ -v
```

Run with coverage reporting:
```bash
python -m pytest tests/ --cov=src/chance_control --cov-report=html
```

### Code Quality

Format code:
```bash
python -m black src/ tests/ examples/
python -m isort src/ tests/ examples/
```

Type checking:
```bash
python -m mypy src/
```

Lint code:
```bash
python -m pyflakes src/
```

## Implementation Notes

### JAX Integration
- All mathematical operations use `jax.numpy` for vectorization
- `eqx.filter_jit` applied to performance-critical functions
- `eqx.filter_vmap` for batch operations over GMM components
- Automatic differentiation via `eqx.filter_grad`

### Optimization Approach
- CVXPy for convex constraint formulation (affine + 2-norm)
- JAXopt integration for unconstrained steps
- Iterative refinement between JAX computations and CVXPy solvers

### Key Differences from Paper
- Uses deterministic feedforward policy (vs. probabilistic in original)
- Simplified terminal constraint implementation for computational efficiency
- Monte Carlo validation instead of theoretical density evolution plots

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{kumagai2024chance,
  title={Chance-Constrained Gaussian Mixture Steering to a Terminal Gaussian Distribution},
  author={Kumagai, Naoya and Oguri, Kenshiro},
  booktitle={2024 IEEE 63rd Conference on Decision and Control (CDC)},
  year={2024},
  organization={IEEE}
}
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow code style**: Run `black`, `isort`, and `mypy` before committing
3. **Add tests**: Ensure new functionality is thoroughly tested
4. **Update documentation**: Include docstrings and update README if needed
5. **Submit a pull request** with a clear description of changes

### Reporting Issues

Please use GitHub Issues to report:
- Bugs or unexpected behavior
- Feature requests or enhancement suggestions
- Documentation improvements
- Performance issues