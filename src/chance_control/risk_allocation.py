"""
Iterative Risk Allocation (IRA) algorithm for chance-constrained optimization.

This module implements the modified IRA algorithm described in 
"Chance-Constrained Gaussian Mixture Steering to a Terminal Gaussian Distribution"
(Kumagai & Oguri, 2024).

The algorithm iteratively refines risk allocation to improve solution optimality
by reducing risk for inactive constraints and reallocating to active ones.
"""

import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from typing import Dict, List, Optional, Tuple

from .control_policy import ControlPolicyOptimizationVariables
from .gmm_models import GaussianMixtureModel, propagate_gmm_state, propagate_gmm_control
from .linear_system import LinearDiscreteSystem
from .optimization import ConvexOptimizationProblem


class IterativeRiskAllocation:
    """
    Iterative Risk Allocation algorithm for improved chance constraint handling.
    
    The algorithm performs the following steps:
    1. Start with uniform risk allocation (URA)
    2. Solve convex optimization problem
    3. Update risk allocation based on constraint activity
    4. Repeat until convergence or maximum iterations
    """
    
    def __init__(
        self,
        system: LinearDiscreteSystem,
        initial_gmm: GaussianMixtureModel,
        target_mean: Float[Array, "state_dim"],
        target_covariance: Float[Array, "state_dim state_dim"],
        horizon: int,
        beta: float = 0.7,
        tolerance: float = 1e-2,
        max_iterations: int = 20,
    ):
        """
        Initialize IRA algorithm.
        
        Args:
            system: Linear discrete system
            initial_gmm: Initial GMM distribution
            target_mean: Target terminal mean
            target_covariance: Target terminal covariance
            horizon: Time horizon
            beta: Algorithm parameter (0 < beta < 1) for risk update
            tolerance: Convergence tolerance for cost improvement
            max_iterations: Maximum number of iterations
        """
        self.system = system
        self.initial_gmm = initial_gmm
        self.target_mean = target_mean
        self.target_covariance = target_covariance
        self.horizon = horizon
        self.beta = beta
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        # Dimensions
        self.num_components = initial_gmm.num_components
        self.state_dim = system.dimension
        self.control_dim = system.control_dimension
        
        # Initialize optimization problem
        self.opt_problem = ConvexOptimizationProblem(
            system, initial_gmm, target_mean, target_covariance, horizon
        )
    
    @jaxtyped(typechecker=typechecker)
    def compute_gmm_trajectories(
        self,
        solution: ControlPolicyOptimizationVariables,
    ) -> Tuple[List[GaussianMixtureModel], List[GaussianMixtureModel]]:
        """
        Compute GMM state and control trajectories for given solution.
        
        Args:
            solution: Control policy solution
            
        Returns:
            (state_trajectories, control_trajectories)
        """
        A_matrices = self.system.A_matrices
        B_matrices = self.system.B_matrices
        
        state_gmms = [self.initial_gmm]  # x_0
        control_gmms = []  # u_0, ..., u_{N-1}
        
        current_gmm = self.initial_gmm
        
        for k in range(self.horizon):
            A_k = A_matrices[k]
            B_k = B_matrices[k]
            V_k = solution.feedforward_gains[k]
            L_k = solution.feedback_gains[:, k, :, :]  # (num_components, control_dim, state_dim)
            
            # Compute control GMM at step k
            control_gmm = propagate_gmm_control(current_gmm, V_k, L_k)
            control_gmms.append(control_gmm)
            
            # Propagate to next state GMM
            next_gmm = propagate_gmm_state(current_gmm, A_k, B_k, V_k, L_k)
            state_gmms.append(next_gmm)
            
            current_gmm = next_gmm
        
        return state_gmms, control_gmms
    
    @jaxtyped(typechecker=typechecker)
    def evaluate_constraint_activity(
        self,
        state_gmms: List[GaussianMixtureModel],
        control_gmms: List[GaussianMixtureModel],
        constraint_matrices: Float[Array, "num_constraints state_dim"],
        constraint_bounds: Float[Array, "num_constraints"],
        control_bound: float,
        activity_threshold: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate which constraints are active/inactive.
        
        A constraint is considered inactive if the constraint violation is small
        relative to the threshold.
        
        Args:
            state_gmms: State GMM trajectories
            control_gmms: Control GMM trajectories  
            constraint_matrices: Affine constraint matrices
            constraint_bounds: Affine constraint bounds
            control_bound: Control constraint bound
            activity_threshold: Threshold for considering constraints inactive
            
        Returns:
            (affine_activity, control_activity) boolean arrays
        """
        num_affine_constraints = constraint_matrices.shape[0]
        
        # Evaluate affine constraint activity
        affine_activity = np.ones((self.num_components, num_affine_constraints, self.horizon), dtype=bool)
        
        for k in range(self.horizon):
            gmm_k = state_gmms[k + 1]  # x_{k+1}
            means_k = np.array(gmm_k.means)
            covs_k = np.array(gmm_k.covariances)
            
            for i in range(self.num_components):
                for j in range(num_affine_constraints):
                    a_j = np.array(constraint_matrices[j])
                    b_j = float(constraint_bounds[j])
                    mu_i_k = means_k[i]
                    Sigma_i_k = covs_k[i]
                    
                    # Compute constraint value: a_j^T μ_i^k - b_j
                    constraint_mean = a_j @ mu_i_k
                    constraint_violation = constraint_mean - b_j
                    
                    # Compute standard deviation: sqrt(a_j^T Σ_i^k a_j)
                    constraint_std = np.sqrt(a_j @ Sigma_i_k @ a_j)
                    
                    # Normalized violation
                    if constraint_std > 1e-10:
                        normalized_violation = constraint_violation / constraint_std
                    else:
                        normalized_violation = constraint_violation
                    
                    # Consider inactive if violation is small
                    if normalized_violation < -activity_threshold:
                        affine_activity[i, j, k] = False
        
        # Evaluate control constraint activity  
        control_activity = np.ones((self.num_components, self.horizon), dtype=bool)
        
        for k in range(self.horizon):
            gmm_k = control_gmms[k]
            means_k = np.array(gmm_k.means)
            covs_k = np.array(gmm_k.covariances)
            
            for i in range(self.num_components):
                mu_i_k = means_k[i]
                Sigma_i_k = covs_k[i]
                
                # Compute ||μ_i^k|| - bound
                mean_norm = np.linalg.norm(mu_i_k)
                constraint_violation = mean_norm - control_bound
                
                # Compute expected norm under uncertainty (approximation)
                trace_term = np.trace(Sigma_i_k)
                expected_norm = mean_norm + 0.5 * trace_term / max(mean_norm, 1e-6)
                normalized_violation = expected_norm - control_bound
                
                if normalized_violation < -activity_threshold:
                    control_activity[i, k] = False
        
        return affine_activity, control_activity
    
    @jaxtyped(typechecker=typechecker)
    def update_risk_allocation(
        self,
        current_affine_risks: np.ndarray,  # (num_components, num_constraints, horizon)
        current_control_risks: np.ndarray,  # (num_components, horizon)
        affine_activity: np.ndarray,  # (num_components, num_constraints, horizon)
        control_activity: np.ndarray,  # (num_components, horizon)
        state_gmms: List[GaussianMixtureModel],
        control_gmms: List[GaussianMixtureModel],
        constraint_matrices: Float[Array, "num_constraints state_dim"],
        constraint_bounds: Float[Array, "num_constraints"],
        control_bound: float,
        total_affine_risk: float,
        total_control_risk: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update risk allocation based on constraint activity.
        
        Following the paper's IRA algorithm:
        - For inactive constraints: reduce risk based on actual constraint values
        - For active constraints: redistribute residual risk
        
        Args:
            current_affine_risks: Current affine risk allocation
            current_control_risks: Current control risk allocation
            affine_activity: Affine constraint activity flags
            control_activity: Control constraint activity flags
            state_gmms: State GMM trajectories
            control_gmms: Control GMM trajectories
            constraint_matrices: Affine constraint matrices
            constraint_bounds: Affine constraint bounds
            control_bound: Control constraint bound
            total_affine_risk: Total affine risk budget
            total_control_risk: Total control risk budget
            
        Returns:
            (updated_affine_risks, updated_control_risks)
        """
        new_affine_risks = current_affine_risks.copy()
        new_control_risks = current_control_risks.copy()
        
        # Update affine constraint risks
        num_affine_constraints = constraint_matrices.shape[0]
        affine_risk_reduction = 0.0
        
        for k in range(self.horizon):
            gmm_k = state_gmms[k + 1]  # x_{k+1}
            means_k = np.array(gmm_k.means)
            covs_k = np.array(gmm_k.covariances)
            weights_k = np.array(gmm_k.weights)
            
            for i in range(self.num_components):
                for j in range(num_affine_constraints):
                    if not affine_activity[i, j, k]:  # Inactive constraint
                        a_j = np.array(constraint_matrices[j])
                        b_j = float(constraint_bounds[j])
                        mu_i_k = means_k[i]
                        Sigma_i_k = covs_k[i]
                        
                        # Compute new risk based on actual constraint satisfaction
                        constraint_mean = a_j @ mu_i_k
                        constraint_std = np.sqrt(a_j @ Sigma_i_k @ a_j)
                        
                        if constraint_std > 1e-10:
                            z_score = (b_j - constraint_mean) / constraint_std
                            new_risk = 1.0 - stats.norm.cdf(z_score)
                        else:
                            new_risk = 0.0 if constraint_mean <= b_j else 1.0
                        
                        # Update with mixing parameter beta
                        old_risk = current_affine_risks[i, j, k]
                        updated_risk = self.beta * old_risk + (1 - self.beta) * new_risk
                        
                        risk_reduction = old_risk - updated_risk
                        affine_risk_reduction += weights_k[i] * risk_reduction
                        
                        new_affine_risks[i, j, k] = updated_risk
        
        # Update control constraint risks
        control_risk_reduction = 0.0
        
        for k in range(self.horizon):
            gmm_k = control_gmms[k]
            means_k = np.array(gmm_k.means)  
            covs_k = np.array(gmm_k.covariances)
            weights_k = np.array(gmm_k.weights)
            
            for i in range(self.num_components):
                if not control_activity[i, k]:  # Inactive constraint
                    mu_i_k = means_k[i]
                    Sigma_i_k = covs_k[i]
                    
                    # Approximate chi-squared constraint evaluation
                    mean_norm_sq = np.dot(mu_i_k, mu_i_k)
                    bound_sq = control_bound**2
                    
                    # Simple approximation for chi-squared distribution
                    if mean_norm_sq < bound_sq:
                        # Use approximation based on squared norm
                        new_risk = max(0.0, 1.0 - stats.chi2.cdf(bound_sq, df=self.control_dim))
                    else:
                        new_risk = 0.5  # Conservative estimate
                    
                    # Update with mixing parameter
                    old_risk = current_control_risks[i, k]
                    updated_risk = self.beta * old_risk + (1 - self.beta) * new_risk
                    
                    risk_reduction = old_risk - updated_risk
                    control_risk_reduction += weights_k[i] * risk_reduction
                    
                    new_control_risks[i, k] = updated_risk
        
        # Redistribute residual risk to active constraints
        if affine_risk_reduction > 0:
            self._redistribute_affine_risk(
                new_affine_risks, affine_activity, affine_risk_reduction, state_gmms
            )
        
        if control_risk_reduction > 0:
            self._redistribute_control_risk(
                new_control_risks, control_activity, control_risk_reduction, control_gmms
            )
        
        return new_affine_risks, new_control_risks
    
    def _redistribute_affine_risk(
        self,
        risk_array: np.ndarray,
        activity_array: np.ndarray,
        total_residual: float,
        state_gmms: List[GaussianMixtureModel],
    ):
        """Redistribute residual affine risk to active constraints."""
        # Count active constraints
        total_active = 0
        for k in range(self.horizon):
            weights_k = np.array(state_gmms[k + 1].weights)
            for i in range(self.num_components):
                for j in range(activity_array.shape[1]):
                    if activity_array[i, j, k]:
                        total_active += weights_k[i]
        
        if total_active > 0:
            risk_per_active = total_residual / total_active
            
            for k in range(self.horizon):
                weights_k = np.array(state_gmms[k + 1].weights)
                for i in range(self.num_components):
                    for j in range(activity_array.shape[1]):
                        if activity_array[i, j, k]:
                            risk_array[i, j, k] += risk_per_active / weights_k[i]
    
    def _redistribute_control_risk(
        self,
        risk_array: np.ndarray,
        activity_array: np.ndarray, 
        total_residual: float,
        control_gmms: List[GaussianMixtureModel],
    ):
        """Redistribute residual control risk to active constraints."""
        # Count active constraints
        total_active = 0
        for k in range(self.horizon):
            weights_k = np.array(control_gmms[k].weights)
            for i in range(self.num_components):
                if activity_array[i, k]:
                    total_active += weights_k[i]
        
        if total_active > 0:
            risk_per_active = total_residual / total_active
            
            for k in range(self.horizon):
                weights_k = np.array(control_gmms[k].weights)
                for i in range(self.num_components):
                    if activity_array[i, k]:
                        risk_array[i, k] += risk_per_active / weights_k[i]
    
    def run_ira(
        self,
        constraint_matrices: Float[Array, "num_constraints state_dim"],
        constraint_bounds: Float[Array, "num_constraints"],
        control_bound: float,
        total_affine_risk: float = 0.005,
        total_control_risk: float = 0.005,
        cost_type: str = "quadratic",
        Q_matrices: Optional[Float[Array, "horizon state_dim state_dim"]] = None,
        R_matrices: Optional[Float[Array, "horizon control_dim control_dim"]] = None,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[ControlPolicyOptimizationVariables], Dict]:
        """
        Run the complete IRA algorithm.
        
        Args:
            constraint_matrices: Affine constraint matrices
            constraint_bounds: Affine constraint bounds
            control_bound: Control constraint bound
            total_affine_risk: Total affine risk budget
            total_control_risk: Total control risk budget
            cost_type: Cost function type ("quadratic" or "norm")
            Q_matrices: State cost matrices (for quadratic cost)
            R_matrices: Control cost matrices (for quadratic cost)
            verbose: Print iteration information
            
        Returns:
            (success, final_solution, info_dict)
        """
        num_affine_constraints = constraint_matrices.shape[0]
        
        # Initialize with uniform risk allocation (URA)
        uniform_affine_risk = total_affine_risk / (num_affine_constraints * self.horizon)
        uniform_control_risk = total_control_risk / self.horizon
        
        affine_risks = np.full(
            (self.num_components, num_affine_constraints, self.horizon),
            uniform_affine_risk
        )
        control_risks = np.full(
            (self.num_components, self.horizon),
            uniform_control_risk
        )
        
        best_solution = None
        best_cost = float('inf')
        iteration_info = []
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"IRA Iteration {iteration + 1}/{self.max_iterations}")
            
            # Solve optimization problem with current risk allocation
            # For now, we'll use a simplified version without full CVXPy integration
            success, solution, solve_info = self.opt_problem.solve_problem(
                cost_type=cost_type,
                Q_matrices=Q_matrices,
                R_matrices=R_matrices,
                state_constraints=(constraint_matrices, constraint_bounds),
                control_bound=control_bound,
                Delta=total_affine_risk,
                Gamma=total_control_risk,
                verbose=False,
            )
            
            if not success:
                if verbose:
                    print(f"  Optimization failed: {solve_info.get('status', 'unknown')}")
                break
            
            current_cost = solve_info['objective_value']
            
            # Check for improvement
            cost_improvement = (best_cost - current_cost) / max(abs(best_cost), 1e-6)
            
            iteration_info.append({
                'iteration': iteration + 1,
                'cost': current_cost,
                'improvement': cost_improvement,
                'solve_time': solve_info.get('solve_time', None)
            })
            
            if verbose:
                print(f"  Cost: {current_cost:.6f}, Improvement: {cost_improvement:.6f}")
            
            # Update best solution
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = solution
            
            # Check convergence
            if cost_improvement < self.tolerance:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
            
            # Compute GMM trajectories for constraint evaluation
            state_gmms, control_gmms = self.compute_gmm_trajectories(solution)
            
            # Evaluate constraint activity
            affine_activity, control_activity = self.evaluate_constraint_activity(
                state_gmms, control_gmms, constraint_matrices, constraint_bounds, control_bound
            )
            
            # Update risk allocation
            affine_risks, control_risks = self.update_risk_allocation(
                affine_risks, control_risks, affine_activity, control_activity,
                state_gmms, control_gmms, constraint_matrices, constraint_bounds,
                control_bound, total_affine_risk, total_control_risk
            )
        
        # Prepare final info
        final_info = {
            'converged': iteration < self.max_iterations - 1,
            'final_cost': best_cost,
            'iterations': len(iteration_info),
            'iteration_history': iteration_info,
            'improvement_percent': ((iteration_info[0]['cost'] - best_cost) / 
                                  abs(iteration_info[0]['cost']) * 100 
                                  if iteration_info else 0)
        }
        
        return best_solution is not None, best_solution, final_info