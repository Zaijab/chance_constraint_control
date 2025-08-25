"""
Chance-constrained control package for Gaussian mixture steering.

This package implements the methodology from "Chance-Constrained Gaussian Mixture 
Steering to a Terminal Gaussian Distribution" (Kumagai & Oguri, 2024).

Key modules:
- linear_system: Linear time-invariant discrete systems
- gmm_models: Gaussian mixture model representation and propagation
- control_policy: Affine feedback control policies with probabilistic selection
- chance_constraints: Deterministic convex reformulation of chance constraints  
- optimization: Convex optimization framework
- risk_allocation: Iterative Risk Allocation (IRA) algorithm

Example usage:
    >>> from chance_control.linear_system import create_double_integrator_2d
    >>> from chance_control.gmm_models import create_paper_initial_gmm
    >>> system = create_double_integrator_2d(horizon=20)
    >>> initial_gmm = create_paper_initial_gmm()
"""

__version__ = "0.1.0"
__author__ = "Implementation based on Kumagai & Oguri (2024)"

# Import main classes and functions
from .linear_system import LinearDiscreteSystem, create_double_integrator_2d
from .gmm_models import (
    GaussianMixtureModel,
    propagate_gmm_state,
    propagate_gmm_control,
    create_paper_initial_gmm,
    create_paper_target_gaussian,
)
from .control_policy import (
    AffineControlPolicy,
    ControlPolicyOptimizationVariables,
    simulate_monte_carlo_trajectories,
)
from .chance_constraints import ChanceConstraintReformulation, create_paper_constraints
from .optimization import ConvexOptimizationProblem, create_paper_cost_matrices
from .risk_allocation import IterativeRiskAllocation

__all__ = [
    # System modeling
    "LinearDiscreteSystem",
    "create_double_integrator_2d",
    
    # GMM handling
    "GaussianMixtureModel", 
    "propagate_gmm_state",
    "propagate_gmm_control",
    "create_paper_initial_gmm",
    "create_paper_target_gaussian",
    
    # Control policies
    "AffineControlPolicy",
    "ControlPolicyOptimizationVariables", 
    "simulate_monte_carlo_trajectories",
    
    # Chance constraints
    "ChanceConstraintReformulation",
    "create_paper_constraints",
    
    # Optimization
    "ConvexOptimizationProblem",
    "create_paper_cost_matrices",
    
    # Risk allocation
    "IterativeRiskAllocation",
]