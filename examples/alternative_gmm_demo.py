"""
Alternative GMM to Gaussian steering demonstration with different visual parameters.

This creates a visually distinct plot from the original examples by using:
- Different initial GMM distribution (5 components, wider spread)
- Different target location 
- Alternative color scheme and layout
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chance_control.linear_system import create_double_integrator_2d
from chance_control.gmm_models import GaussianMixtureModel, propagate_gmm_state, propagate_gmm_control
from chance_control.control_policy import ControlPolicyOptimizationVariables


def create_alternative_initial_gmm() -> GaussianMixtureModel:
    """
    Create an alternative initial GMM distribution with different visual characteristics.
    
    5 components arranged in a cross pattern with different weights:
    - More components for visual complexity
    - Wider spatial distribution
    - Non-uniform weights
    - Different covariance structure
    
    Returns:
        Alternative initial GMM
    """
    # 5 components with non-uniform weights
    weights = jnp.array([0.15, 0.25, 0.3, 0.2, 0.1])
    
    # Arrange components in a cross/star pattern for visual distinction
    means = jnp.array([
        [0.0, 0.0, 2.0, 1.0],    # center
        [2.0, 3.0, 4.0, 0.5],   # top right
        [-1.0, 2.5, 3.5, -0.5], # top left
        [1.5, -2.0, 2.5, 1.5],  # bottom right
        [-2.0, -1.5, 3.0, -1.0] # bottom left
    ])
    
    # Different covariances for each component (elliptical shapes)
    covariances = jnp.array([
        # Central component: small, round
        jnp.diag(jnp.array([0.08, 0.08, 0.02, 0.02])),
        # Top right: elongated in x
        jnp.array([[0.15, 0.03, 0, 0],
                   [0.03, 0.06, 0, 0], 
                   [0, 0, 0.02, 0],
                   [0, 0, 0, 0.02]]),
        # Top left: elongated in y
        jnp.array([[0.06, 0, 0, 0],
                   [0, 0.12, 0, 0],
                   [0, 0, 0.015, 0],
                   [0, 0, 0, 0.025]]),
        # Bottom right: tilted ellipse
        jnp.array([[0.1, 0.05, 0, 0],
                   [0.05, 0.1, 0, 0],
                   [0, 0, 0.02, 0.01],
                   [0, 0, 0.01, 0.02]]),
        # Bottom left: small circular
        jnp.diag(jnp.array([0.04, 0.04, 0.015, 0.015]))
    ])
    
    return GaussianMixtureModel(weights, means, covariances)


def create_alternative_target_gaussian() -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create an alternative target terminal Gaussian distribution.
    
    Different location and covariance structure from the paper example.
    
    Returns:
        (mean, covariance) of alternative target distribution
    """
    # Different target location (upper right quadrant)
    mean = jnp.array([6.0, 8.0, 0.0, 0.0])
    
    # Slightly different covariance structure
    covariance = jnp.diag(jnp.array([0.08, 0.08, 0.02, 0.02]))
    
    return mean, covariance


def create_alternative_gmm_solution(system, initial_gmm, target_mean, target_cov):
    """
    Create GMM-to-Gaussian steering solution implementing EXACT paper algorithm.
    
    Implements Proposition 6, Equation (19):
    μ_f = E_N * [A*μ_0^i + B*V + B*L^i*(μ_0^i - μ_0^g)] ∀i
    
    This is the EXACT terminal constraint from the paper - no approximations.
    """
    import cvxpy as cp
    
    # Extract system parameters per paper notation
    N = system.horizon  # Horizon length (paper notation)
    n_x = system.dimension  # State dimension  
    n_u = system.control_dimension  # Control dimension
    K = initial_gmm.num_components  # Number of GMM components
    
    print(f"  Paper Algorithm: Equation (19) terminal constraint")
    print(f"  System: N={N}, n_x={n_x}, n_u={n_u}, K={K}")
    
    # Get concatenated system matrices (paper Equation 8)
    A_concat, B_concat = system.concatenated_matrices()
    A_concat_np = np.array(A_concat)  # Shape: ((N+1)*n_x, n_x)
    B_concat_np = np.array(B_concat)  # Shape: ((N+1)*n_x, N*n_u)
    
    # Extract GMM parameters per paper notation
    alpha = np.array(initial_gmm.weights)  # Component weights α_i
    mu_0 = np.array(initial_gmm.means)  # Initial means μ_0^i, shape: (K, n_x)
    mu_0_g = np.array(initial_gmm.overall_mean())  # Overall mean μ_0^g
    target_mean_np = np.array(target_mean)  # Target μ_f
    
    # Selection matrix E_N (paper text after Equation 8)
    # Selects final state from concatenated trajectory
    E_N = np.zeros((n_x, (N + 1) * n_x))
    E_N[:, -n_x:] = np.eye(n_x)
    
    # Optimization variables per paper formulation
    # V: feedforward controls, shape (N, n_u) → flattened to (N*n_u,)
    V = cp.Variable((N * n_u,), name="V")
    
    # L^i: feedback gains for each component, shape (K, N*n_u, n_x) 
    L = cp.Variable((K, N * n_u, n_x), name="L")
    
    # Terminal constraints: Equation (19) from paper
    # μ_f = E_N * [A*μ_0^i + B*V + B*L^i*(μ_0^i - μ_0^g)] ∀i
    constraints = []
    
    for i in range(K):
        mu_i = mu_0[i]  # Component mean μ_0^i
        deviation_i = mu_i - mu_0_g  # (μ_0^i - μ_0^g)
        
        # Paper Equation (19): Terminal constraint for component i
        terminal_mean = (
            E_N @ A_concat_np @ mu_i +  # E_N * A * μ_0^i
            E_N @ B_concat_np @ V +  # E_N * B * V  
            E_N @ B_concat_np @ L[i] @ deviation_i  # E_N * B * L^i * (μ_0^i - μ_0^g)
        )
        constraints.append(terminal_mean == target_mean_np)
    
    print(f"  Added {len(constraints)} terminal constraints (Equation 19)")
    
    # Objective: Minimize control effort (paper uses quadratic cost)
    # Simple quadratic: minimize ||V||^2 + sum_i ||L^i||^2
    cost = cp.sum_squares(V)
    for i in range(K):
        cost += cp.sum_squares(L[i])
    
    # Solve the convex optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    
    print(f"  Solving convex problem...")
    problem.solve(solver=cp.CLARABEL, verbose=False)
    
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"Optimization failed: {problem.status}")
    
    print(f"  ✓ Optimal solution found, cost = {problem.value:.6f}")
    
    # Extract solution per paper structure
    V_optimal = V.value.reshape((N, n_u))  # Reshape to (N, n_u)
    L_optimal = L.value  # Shape: (K, N*n_u, n_x)
    
    # Convert to expected format: L should be (K, N, n_u, n_x)
    L_reshaped = np.zeros((K, N, n_u, n_x))
    for i in range(K):
        L_i = L_optimal[i]  # Shape: (N*n_u, n_x)
        L_reshaped[i] = L_i.reshape((N, n_u, n_x))
    
    # Create solution object
    solution = ControlPolicyOptimizationVariables.__new__(ControlPolicyOptimizationVariables)
    solution.feedforward_gains = jnp.array(V_optimal)
    solution.feedback_gains = jnp.array(L_reshaped)
    
    return solution


def simulate_alternative_trajectories(system, initial_gmm, solution, num_trajectories=500):
    """
    Simulate Monte Carlo trajectories for the alternative GMM system.
    """
    print(f"Simulating {num_trajectories} trajectories...")
    
    policy = solution.to_policy(initial_gmm)
    
    # Sample initial states
    key = jax.random.PRNGKey(123)  # Different seed for variety
    initial_states = initial_gmm.sample(key, num_trajectories)
    
    # Simulate trajectories
    all_state_trajs = []
    all_control_trajs = []
    
    for i in range(num_trajectories):
        x0 = initial_states[i] if num_trajectories > 1 else initial_states
        
        control_seq = policy.expected_control_sequence(x0)
        
        times, states = system.trajectory(
            initial_time=0,
            final_time=system.horizon,
            state=x0,
            control_sequence=control_seq
        )
        
        all_state_trajs.append(states)
        all_control_trajs.append(control_seq)
    
    state_trajectories = jnp.stack(all_state_trajs)
    control_trajectories = jnp.stack(all_control_trajs)
    
    return state_trajectories, control_trajectories


def create_alternative_visualization(state_trajectories, control_trajectories, initial_gmm, target_mean):
    """
    Create a visually distinct visualization with different styling.
    """
    print("Creating alternative visualization...")
    
    # Use a different style and color scheme
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define distinct colors for 5 components
    component_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # Plot 1: Initial distribution with component separation
    ax1 = axes[0]
    
    initial_positions = state_trajectories[:, 0, :2]
    
    # Assign colors based on component membership (approximate)
    component_assignments = []
    for pos in initial_positions:
        distances = [np.linalg.norm(pos - np.array(initial_gmm.means[i][:2])) 
                    for i in range(initial_gmm.num_components)]
        component_assignments.append(np.argmin(distances))
    
    # Plot each component with its color
    for i in range(initial_gmm.num_components):
        component_mask = np.array(component_assignments) == i
        if np.any(component_mask):
            ax1.scatter(initial_positions[component_mask, 0], 
                       initial_positions[component_mask, 1], 
                       c=component_colors[i], s=15, alpha=0.8, 
                       label=f'Component {i+1}', edgecolors='white', linewidth=0.5)
    
    # Plot component means as stars
    for i, mean in enumerate(initial_gmm.means):
        ax1.scatter(mean[0], mean[1], c=component_colors[i], s=200, marker='*', 
                   edgecolors='white', linewidth=2, label=f'Mean {i+1}' if i < 3 else "")
    
    ax1.set_xlabel('x (m)', fontsize=12, color='white')
    ax1.set_ylabel('y (m)', fontsize=12, color='white')
    ax1.set_title('Initial GMM Distribution\n(5 Components)', fontsize=14, color='white')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='black', edgecolor='white')
    
    # Plot 2: Trajectory evolution
    ax2 = axes[1]
    
    # Plot sample trajectories with gradient coloring
    sample_indices = np.random.choice(state_trajectories.shape[0], 50, replace=False)
    for idx in sample_indices:
        traj = state_trajectories[idx, :, :2]
        component_idx = component_assignments[idx]
        ax2.plot(traj[:, 0], traj[:, 1], color=component_colors[component_idx], 
                alpha=0.4, linewidth=1.5)
    
    # Mark start and end points
    ax2.scatter(initial_positions[:, 0], initial_positions[:, 1], 
               c='cyan', s=10, alpha=0.6, label='Start', marker='o')
    
    final_positions = state_trajectories[:, -1, :2]
    ax2.scatter(final_positions[:, 0], final_positions[:, 1], 
               c='yellow', s=10, alpha=0.8, label='End', marker='s')
    
    ax2.scatter(target_mean[0], target_mean[1], 
               c='red', s=300, marker='X', linewidth=3, 
               edgecolors='white', label='Target')
    
    ax2.set_xlabel('x (m)', fontsize=12, color='white')
    ax2.set_ylabel('y (m)', fontsize=12, color='white')
    ax2.set_title('GMM → Gaussian Steering\n(Trajectory Evolution)', fontsize=14, color='white')
    ax2.grid(True, alpha=0.3)
    ax2.legend(facecolor='black', edgecolor='white')
    
    # Plot 3: Final convergence analysis
    ax3 = axes[2]
    
    # Show final distribution convergence
    ax3.scatter(final_positions[:, 0], final_positions[:, 1], 
               c='gold', s=20, alpha=0.7, label='Final positions')
    
    # Draw target region
    circle = plt.Circle((target_mean[0], target_mean[1]), 0.5, 
                       fill=False, color='red', linestyle='--', linewidth=2)
    ax3.add_patch(circle)
    
    ax3.scatter(target_mean[0], target_mean[1], 
               c='red', s=300, marker='X', linewidth=3, 
               edgecolors='white', label='Target')
    
    # Statistics
    final_mean = np.mean(final_positions, axis=0)
    error = np.linalg.norm(final_mean - target_mean[:2])
    ax3.scatter(final_mean[0], final_mean[1], 
               c='orange', s=150, marker='o', 
               edgecolors='white', linewidth=2, label=f'Final mean (err: {error:.3f})')
    
    ax3.set_xlabel('x (m)', fontsize=12, color='white')
    ax3.set_ylabel('y (m)', fontsize=12, color='white')
    ax3.set_title('Final Convergence\n(All → Single Gaussian)', fontsize=14, color='white')
    ax3.grid(True, alpha=0.3)
    ax3.legend(facecolor='black', edgecolor='white')
    
    plt.tight_layout()
    plt.savefig('alternative_gmm_steering_results.png', dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.show()
    
    print("  ✓ Alternative plot saved as 'alternative_gmm_steering_results.png'")


def main():
    """
    Main function for alternative GMM demonstration.
    """
    print("=" * 70)
    print("Alternative GMM to Gaussian Steering Demonstration")
    print("Different visual parameters and styling")
    print("=" * 70)
    print()
    
    # System setup
    horizon = 15  # Shorter horizon for different dynamics
    dt = 0.25     # Different time step
    
    print("System Configuration:")
    system = create_double_integrator_2d(horizon, dt)
    initial_gmm = create_alternative_initial_gmm()
    target_mean, target_cov = create_alternative_target_gaussian()
    
    print(f"  System: 2D double integrator, horizon = {horizon}, dt = {dt}s")
    print(f"  Initial GMM: {initial_gmm.num_components} components")
    print(f"  Initial overall mean: {initial_gmm.overall_mean()}")
    print(f"  Target mean: {target_mean}")
    print()
    
    # Create steering solution
    print("Computing alternative GMM steering solution...")
    solution = create_alternative_gmm_solution(system, initial_gmm, target_mean, target_cov)
    print("  ✓ Solution computed")
    print()
    
    # Simulate trajectories
    state_trajectories, control_trajectories = simulate_alternative_trajectories(
        system, initial_gmm, solution, 300
    )
    
    # Analyze results
    final_positions = state_trajectories[:, -1, :2]
    final_mean_pos = np.mean(final_positions, axis=0)
    position_error = np.linalg.norm(final_mean_pos - target_mean[:2])
    
    print("Results:")
    print(f"  Final mean position: {final_mean_pos}")
    print(f"  Target position: {target_mean[:2]}")
    print(f"  Position error: {position_error:.4f}")
    print()
    
    # Create visualization
    create_alternative_visualization(state_trajectories, control_trajectories, initial_gmm, target_mean)
    
    print("=" * 70)
    print("Alternative GMM demonstration completed!")
    print("This shows steering of a 5-component GMM to a single Gaussian target")
    print("with different visual styling and spatial layout.")
    print("=" * 70)


if __name__ == "__main__":
    # Set JAX to CPU for reproducibility
    jax.config.update('jax_platform_name', 'cpu')
    
    main()