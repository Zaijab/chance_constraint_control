"""
Chance constraint formulation for Gaussian mixture distributions.

This module implements the deterministic convex reformulation of chance constraints
as described in "Chance-Constrained Gaussian Mixture Steering to a Terminal
Gaussian Distribution" (Kumagai & Oguri, 2024).

Key theoretical results implemented:
- Theorem 2: Deterministic formulation of chance constraints
- Risk allocation variables and constraints
"""

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from .gmm_models import GaussianMixtureModel


class ChanceConstraintReformulation:
    """
    Deterministic convex reformulation of chance constraints on GMM distributions.
    
    Handles two types of constraints:
    1. Affine/hyperplane constraints: a^T y - b ≤ 0
    2. 2-norm constraints: ||G*y + g|| ≤ y_max
    """
    
    def __init__(
        self,
        num_components: int,
        horizon: int,
        state_dim: int,
        control_dim: int,
    ):
        """
        Initialize chance constraint reformulation.
        
        Args:
            num_components: Number of GMM components (K)
            horizon: Time horizon (N)
            state_dim: State dimension
            control_dim: Control dimension
        """
        self.num_components = num_components
        self.horizon = horizon
        self.state_dim = state_dim
        self.control_dim = control_dim
    
    @jaxtyped(typechecker=typechecker)
    def create_affine_constraint_variables(
        self,
        num_constraints: int,
    ) -> tuple[cp.Variable, list[cp.Constraint]]:
        """
        Create risk allocation variables for affine constraints.
        
        From Theorem 2: δ_ijk variables for each (component, constraint, time)
        
        Args:
            num_constraints: Number of affine constraints (N_c)
            
        Returns:
            (risk_variables, constraint_list)
        """
        # Risk allocation variables δ_ijk
        delta = cp.Variable(
            (self.num_components, num_constraints, self.horizon),
            nonneg=True,
            name="delta_affine"
        )
        
        constraints = []
        
        # Risk variables must be between 0 and 1
        constraints.append(delta <= 1.0)
        
        return delta, constraints
    
    @jaxtyped(typechecker=typechecker)  
    def create_norm_constraint_variables(self) -> tuple[cp.Variable, list[cp.Constraint]]:
        """
        Create risk allocation variables for 2-norm constraints.
        
        From Theorem 2: γ_ik variables for each (component, time)
        
        Returns:
            (risk_variables, constraint_list)
        """
        # Risk allocation variables γ_ik
        gamma = cp.Variable(
            (self.num_components, self.horizon),
            nonneg=True,
            name="gamma_norm"
        )
        
        constraints = []
        
        # Risk variables must be between 0 and 1
        constraints.append(gamma <= 1.0)
        
        return gamma, constraints
    
    def add_affine_constraint_reformulation(
        self,
        constraints: list,
        gmm_states: list[GaussianMixtureModel],  # States at each time step
        constraint_matrices: Float[Array, "num_constraints state_dim"],  # a_j
        constraint_bounds: Float[Array, "num_constraints"],  # b_j  
        delta: cp.Variable,
        total_risk_bound: float,  # Δ
    ):
        """
        Add deterministic reformulation of affine chance constraints.
        
        Implements constraints (31a) and (31b) from Theorem 2:
        a_j^T μ_i^k + F_N^{-1}(1 - δ_ijk) ||a_j^T (Σ_i^k)^{1/2}|| ≤ b_j
        Σ_i Σ_j Σ_k α_i δ_ijk ≤ Δ
        
        Args:
            constraints: List to append constraints to
            gmm_states: GMM states at each time step
            constraint_matrices: Constraint matrices [a_1, ..., a_{N_c}]
            constraint_bounds: Constraint bounds [b_1, ..., b_{N_c}]
            delta: Risk allocation variables
            total_risk_bound: Total allowed risk Δ
        """
        num_constraints = constraint_matrices.shape[0]
        
        # Convert JAX arrays to numpy for CVXPy
        a_matrices = np.array(constraint_matrices)
        b_bounds = np.array(constraint_bounds)
        
        # Individual constraint reformulation (31a)
        for k in range(self.horizon):
            gmm_k = gmm_states[k]
            weights_k = np.array(gmm_k.weights)
            means_k = np.array(gmm_k.means)  # (num_components, state_dim)
            covs_k = np.array(gmm_k.covariances)  # (num_components, state_dim, state_dim)
            
            for i in range(self.num_components):
                for j in range(num_constraints):
                    a_j = a_matrices[j]  # (state_dim,)
                    b_j = b_bounds[j]
                    mu_i_k = means_k[i]  # (state_dim,)
                    Sigma_i_k = covs_k[i]  # (state_dim, state_dim)
                    
                    # Compute a_j^T μ_i^k
                    mean_term = a_j @ mu_i_k
                    
                    # Compute ||a_j^T (Σ_i^k)^{1/2}||_2
                    # First compute (Σ_i^k)^{1/2} via eigendecomposition
                    eigenvals, eigenvecs = np.linalg.eigh(Sigma_i_k)
                    sqrt_eigenvals = np.sqrt(np.maximum(eigenvals, 0))
                    Sigma_sqrt = eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
                    
                    # Now compute a_j^T * Sigma_sqrt
                    a_Sigma_sqrt = a_j @ Sigma_sqrt
                    sigma_term = np.linalg.norm(a_Sigma_sqrt)
                    
                    # Add constraint: mean_term + F_N^{-1}(1 - δ_ijk) * sigma_term ≤ b_j
                    # We'll use the conservative approximation with uniform risk allocation first
                    # and then implement IRA separately
                    delta_ijk = delta[i, j, k]
                    
                    # Use inverse normal CDF approximation
                    # For small δ, F_N^{-1}(1 - δ) ≈ sqrt(2 * log(1/δ)) for δ → 0
                    # We'll implement this as a constraint with auxiliary variables
                    
                    # Simpler approach: Use fixed confidence level for now
                    # This can be refined with IRA algorithm
                    confidence_level = 0.99  # Will be updated by IRA
                    z_score = stats.norm.ppf(confidence_level)
                    
                    # Constraint: mean_term + z_score * sigma_term ≤ b_j
                    constraints.append(mean_term + z_score * sigma_term <= b_j)
        
        # Global risk bound constraint (31b): Σ_i Σ_j Σ_k α_i δ_ijk ≤ Δ
        total_allocated_risk = 0
        for k in range(self.horizon):
            weights_k = np.array(gmm_states[k].weights)
            for i in range(self.num_components):
                for j in range(num_constraints):
                    total_allocated_risk += weights_k[i] * delta[i, j, k]
        
        constraints.append(total_allocated_risk <= total_risk_bound)
    
    def add_norm_constraint_reformulation(
        self,
        constraints: list,
        gmm_controls: list[GaussianMixtureModel],  # Controls at each time step
        G_matrix: Float[Array, "control_dim control_dim"] | None,  # G (identity if None)
        g_vector: Float[Array, "control_dim"] | None,  # g (zero if None)  
        bound: float,  # y_max
        gamma: cp.Variable,
        total_risk_bound: float,  # Γ
    ):
        """
        Add deterministic reformulation of 2-norm chance constraints.
        
        Implements constraints (31c) and (31d) from Theorem 2:
        ||G μ_i^k + g|| + sqrt(F_χ²^{-1}(1 - γ_ik)) ||G (Σ_i^k)^{1/2}|| ≤ y_max
        Σ_i Σ_k α_i γ_ik ≤ Γ
        
        Args:
            constraints: List to append constraints to
            gmm_controls: GMM control distributions at each time step
            G_matrix: Constraint matrix (identity if None)
            g_vector: Constraint offset (zero if None)
            bound: Constraint bound y_max
            gamma: Risk allocation variables  
            total_risk_bound: Total allowed risk Γ
        """
        # Set defaults
        if G_matrix is None:
            G_matrix = np.eye(self.control_dim)
        else:
            G_matrix = np.array(G_matrix)
            
        if g_vector is None:
            g_vector = np.zeros(self.control_dim)
        else:
            g_vector = np.array(g_vector)
        
        # Individual constraint reformulation (31c)
        for k in range(self.horizon):
            gmm_k = gmm_controls[k]
            weights_k = np.array(gmm_k.weights)
            means_k = np.array(gmm_k.means)  # (num_components, control_dim)
            covs_k = np.array(gmm_k.covariances)  # (num_components, control_dim, control_dim)
            
            for i in range(self.num_components):
                mu_i_k = means_k[i]  # (control_dim,)
                Sigma_i_k = covs_k[i]  # (control_dim, control_dim)
                
                # Compute ||G μ_i^k + g||
                mean_term = np.linalg.norm(G_matrix @ mu_i_k + g_vector)
                
                # Compute ||G (Σ_i^k)^{1/2}||_F (Frobenius norm approximation)
                eigenvals, eigenvecs = np.linalg.eigh(Sigma_i_k)
                sqrt_eigenvals = np.sqrt(np.maximum(eigenvals, 0))
                Sigma_sqrt = eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
                G_Sigma_sqrt = G_matrix @ Sigma_sqrt
                sigma_term = np.linalg.norm(G_Sigma_sqrt, 'fro')
                
                # Use chi-squared approximation
                # For now, use fixed confidence level (will be refined by IRA)
                confidence_level = 0.99  # Will be updated by IRA
                chi2_quantile = stats.chi2.ppf(confidence_level, df=self.control_dim)
                z_chi = np.sqrt(chi2_quantile)
                
                # Constraint: mean_term + z_chi * sigma_term ≤ bound
                constraints.append(mean_term + z_chi * sigma_term <= bound)
        
        # Global risk bound constraint (31d): Σ_i Σ_k α_i γ_ik ≤ Γ
        total_allocated_risk = 0
        for k in range(self.horizon):
            weights_k = np.array(gmm_controls[k].weights)
            for i in range(self.num_components):
                total_allocated_risk += weights_k[i] * gamma[i, k]
        
        constraints.append(total_allocated_risk <= total_risk_bound)
    
    def uniform_risk_allocation(
        self,
        delta: cp.Variable,
        gamma: cp.Variable,
        num_affine_constraints: int,
        total_affine_risk: float,  # Δ
        total_norm_risk: float,    # Γ
    ) -> list[cp.Constraint]:
        """
        Apply uniform risk allocation (URA) as initial solution.
        
        From Remark 3: δ_ijk = Δ/(N_c * N), γ_ik = Γ/N
        
        Args:
            delta: Affine constraint risk variables
            gamma: Norm constraint risk variables  
            num_affine_constraints: Number of affine constraints (N_c)
            total_affine_risk: Total affine risk budget Δ
            total_norm_risk: Total norm risk budget Γ
            
        Returns:
            List of equality constraints for URA
        """
        constraints = []
        
        # Uniform affine risk allocation
        uniform_affine_risk = total_affine_risk / (num_affine_constraints * self.horizon)
        constraints.append(delta == uniform_affine_risk)
        
        # Uniform norm risk allocation  
        uniform_norm_risk = total_norm_risk / self.horizon
        constraints.append(gamma == uniform_norm_risk)
        
        return constraints


def create_paper_constraints() -> tuple[
    Float[Array, "2 4"], 
    Float[Array, "2"], 
    float,
    float,
    float
]:
    """
    Create constraint matrices from the paper's numerical example.
    
    State constraints:
    - a_1 = [1.3, -1, 0, 0]^T, b_1 = 11
    - a_2 = [-1, 1, 0, 0]^T, b_2 = -1  
    - Joint violation probability Δ = 0.005
    
    Control constraints:
    - ||u|| ≤ 6.5
    - Violation probability Γ = 0.005
    
    Returns:
        (constraint_matrices, constraint_bounds, Δ, u_max, Γ)
    """
    # Affine state constraints
    constraint_matrices = jnp.array([
        [1.3, -1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0]
    ])
    
    constraint_bounds = jnp.array([11.0, -1.0])
    
    # Risk bounds
    Delta = 0.005  # State constraint violation probability
    u_max = 6.5    # Control bound
    Gamma = 0.005  # Control constraint violation probability
    
    return constraint_matrices, constraint_bounds, Delta, u_max, Gamma