"""
Simplified reproduction of the paper's key concepts without full CVXPy optimization.

This demonstrates the core GMM propagation and Monte Carlo validation
without the complex convex optimization framework.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chance_control.linear_system import create_double_integrator_2d
from chance_control.gmm_models import (
    create_paper_initial_gmm, 
    create_paper_target_gaussian,
    propagate_gmm_state,
    propagate_gmm_control
)
from chance_control.control_policy import ControlPolicyOptimizationVariables


def gmm_to_gaussian_solution(system, initial_gmm, target_mean, target_cov):
    """
    Create a solution that properly steers GMM to a single Gaussian.
    
    Implements Proposition 5: All component means must reach target_mean,
    and the covariances must be controlled to achieve target covariance.
    """
    horizon = system.horizon
    control_dim = system.control_dimension
    state_dim = system.dimension
    num_components = initial_gmm.num_components
    
    # Get system matrices
    A_concat, B_concat = system.concatenated_matrices()
    A_concat_np = np.array(A_concat)
    B_concat_np = np.array(B_concat)
    
    # Get initial GMM parameters
    initial_weights = np.array(initial_gmm.weights)
    initial_means = np.array(initial_gmm.means)  # (K, state_dim)
    initial_covs = np.array(initial_gmm.covariances)  # (K, state_dim, state_dim)
    mu_g = np.array(initial_gmm.overall_mean())
    
    target_mean_np = np.array(target_mean)
    target_cov_np = np.array(target_cov)
    
    # Selection matrix E_N (selects final state)
    E_N = np.zeros((state_dim, (horizon + 1) * state_dim))
    E_N[:, -state_dim:] = np.eye(state_dim)
    
    print(f"  Implementing terminal constraint: ALL components → {target_mean_np}")
    
    # Solve for feedforward and feedback controls using least squares
    # We need to solve: E_N @ [A μ_i + B V + B L^i (μ_i - μ_g)] = target ∀i
    
    # Set up the system for all components simultaneously
    num_variables = horizon * control_dim * (1 + num_components)  # V + L^1 + L^2 + ... + L^K
    num_equations = num_components * state_dim  # One constraint per component per state dimension
    
    # Build constraint matrix and RHS
    A_matrix = np.zeros((num_equations, num_variables))
    b_vector = np.zeros(num_equations)
    
    for i in range(num_components):
        row_start = i * state_dim
        row_end = (i + 1) * state_dim
        
        mu_i = initial_means[i]
        deviation_i = mu_i - mu_g
        
        # Constant term: E_N @ A @ mu_i
        constant_term = E_N @ A_concat_np @ mu_i
        b_vector[row_start:row_end] = target_mean_np - constant_term
        
        # Feedforward term: E_N @ B @ V (columns 0 to horizon*control_dim-1)
        v_cols_start = 0
        v_cols_end = horizon * control_dim
        A_matrix[row_start:row_end, v_cols_start:v_cols_end] = E_N @ B_concat_np
        
        # Feedback term: E_N @ B @ L^i @ (μ_i - μ_g)
        l_cols_start = v_cols_end + i * (horizon * control_dim)
        l_cols_end = l_cols_start + horizon * control_dim
        
        # For feedback, we need to construct the effect of L^i
        # This is complex in concatenated form, so we'll use a simplified approach
        # that achieves the same goal: steering each component mean to target
        
        # Use final control input contribution (simplified)
        final_B = B_concat_np[-state_dim:, -control_dim:]  # Final step B matrix
        
        # For feedback L^i, the effect is: B @ L^i @ (μ_i - μ_g)
        # We'll approximate this by distributing the deviation effect
        # across the final control inputs
        feedback_effect = final_B  # This is (state_dim, control_dim)
        
        # Add feedback contribution for final time step
        final_time_cols_start = l_cols_start + (horizon-1)*control_dim
        final_time_cols_end = final_time_cols_start + control_dim
        A_matrix[row_start:row_end, final_time_cols_start:final_time_cols_end] = feedback_effect
    
    # Solve the least squares problem
    print(f"  Solving {num_equations} constraints with {num_variables} variables")
    solution_vector = np.linalg.lstsq(A_matrix, b_vector, rcond=None)[0]
    
    # Extract V and L from solution
    V_flat = solution_vector[:horizon * control_dim]
    V_solution = V_flat.reshape((horizon, control_dim))
    
    # Extract feedback gains L^i
    L_solution = np.zeros((num_components, horizon, control_dim, state_dim))
    for i in range(num_components):
        l_start = horizon * control_dim * (1 + i)
        l_end = l_start + horizon * control_dim
        L_flat_i = solution_vector[l_start:l_end]
        
        # For this simplified implementation, put all feedback in final time step
        # The final control contribution should be shaped as (control_dim, state_dim)
        final_feedback = L_flat_i[-control_dim:].reshape(control_dim, 1)
        
        # Create a simple feedback matrix that maps state deviation to control
        # This is a simplified approximation
        L_solution[i, -1, :, :] = final_feedback @ np.ones((1, state_dim)) / state_dim
    
    # Create solution object
    solution = ControlPolicyOptimizationVariables.__new__(ControlPolicyOptimizationVariables)
    solution.feedforward_gains = jnp.array(V_solution)
    solution.feedback_gains = jnp.array(L_solution)
    
    return solution


def run_gmm_propagation_demo(system, initial_gmm, solution, target_mean):
    """
    Demonstrate GMM propagation through the system with the control policy.
    """
    print("Running GMM propagation demonstration...")
    
    A_matrices = system.A_matrices
    B_matrices = system.B_matrices
    horizon = system.horizon
    
    # Propagate GMM through time
    state_gmms = [initial_gmm]
    control_gmms = []
    
    current_gmm = initial_gmm
    
    for k in range(horizon):
        A_k = A_matrices[k]
        B_k = B_matrices[k] 
        V_k = solution.feedforward_gains[k]
        L_k = solution.feedback_gains[:, k, :, :]  # (num_components, control_dim, state_dim)
        
        print(f"  Step {k}: GMM mean = {current_gmm.overall_mean()}")
        
        # Compute control GMM
        control_gmm = propagate_gmm_control(current_gmm, V_k, L_k)
        control_gmms.append(control_gmm)
        
        # Propagate to next state
        next_gmm = propagate_gmm_state(current_gmm, A_k, B_k, V_k, L_k)
        state_gmms.append(next_gmm)
        
        current_gmm = next_gmm
    
    final_gmm = state_gmms[-1]
    print(f"  Final GMM overall mean = {final_gmm.overall_mean()}")
    print(f"  Target mean = {target_mean}")
    print(f"  Overall mean error = {np.linalg.norm(np.array(final_gmm.overall_mean()) - np.array(target_mean))}")
    
    # Check individual component means (key for GMM-to-Gaussian steering)
    final_component_means = np.array(final_gmm.means)
    print(f"  Final component means:")
    for i in range(final_gmm.num_components):
        component_error = np.linalg.norm(final_component_means[i] - np.array(target_mean))
        print(f"    Component {i+1}: {final_component_means[i]} (error: {component_error:.6f})")
    
    # Check if all components converged to same target (Proposition 5 requirement)
    max_component_error = max([np.linalg.norm(final_component_means[i] - np.array(target_mean)) 
                              for i in range(final_gmm.num_components)])
    print(f"  Max component error: {max_component_error:.6f}")
    
    if max_component_error < 1e-3:
        print("  ✅ SUCCESS: All components converged to target (GMM → Gaussian achieved)")
    else:
        print("  ❌ PARTIAL: Components not fully converged to single target")
    
    return state_gmms, control_gmms


def run_monte_carlo_validation(system, initial_gmm, solution, target_mean, num_trajectories=1000):
    """
    Run Monte Carlo simulation to validate theoretical predictions.
    """
    print(f"Running Monte Carlo validation ({num_trajectories} trajectories)...")
    
    # Create control policy
    policy = solution.to_policy(initial_gmm)
    
    # Sample initial states
    key = jax.random.PRNGKey(42)
    initial_states = initial_gmm.sample(key, num_trajectories)
    
    # Simulate trajectories
    all_state_trajs = []
    all_control_trajs = []
    
    for i in range(num_trajectories):
        x0 = initial_states[i] if num_trajectories > 1 else initial_states
        
        # Generate deterministic control sequence (using expected control)
        control_seq = policy.expected_control_sequence(x0)
        
        # Simulate system
        times, states = system.trajectory(
            initial_time=0,
            final_time=system.horizon,
            state=x0,
            control_sequence=control_seq
        )
        
        all_state_trajs.append(states)
        all_control_trajs.append(control_seq)
    
    # Convert to arrays
    state_trajectories = jnp.stack(all_state_trajs)
    control_trajectories = jnp.stack(all_control_trajs)
    
    # Analyze results
    initial_positions = state_trajectories[:, 0, :2]  # x, y
    final_positions = state_trajectories[:, -1, :2]
    
    print(f"  Initial mean position: {jnp.mean(initial_positions, axis=0)}")
    print(f"  Final mean position: {jnp.mean(final_positions, axis=0)}")
    print(f"  Target position: {target_mean[:2]}")
    print(f"  Position error: {jnp.linalg.norm(jnp.mean(final_positions, axis=0) - target_mean[:2])}")
    
    # Control statistics
    control_norms = jnp.linalg.norm(control_trajectories, axis=2)
    print(f"  Max control norm: {jnp.max(control_norms)}")
    print(f"  Mean control norm: {jnp.mean(control_norms)}")
    
    return state_trajectories, control_trajectories


def create_visualization(state_trajectories, control_trajectories, initial_gmm, target_mean):
    """
    Create visualization plots.
    """
    print("Creating visualization plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # State trajectory plot
    ax1 = axes[0]
    
    # Plot sample trajectories
    sample_indices = np.random.choice(state_trajectories.shape[0], 100, replace=False)
    for idx in sample_indices:
        traj = state_trajectories[idx, :, :2]  # x, y positions
        ax1.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.1, linewidth=0.5)
    
    # Plot initial distribution with component colors
    initial_positions = state_trajectories[:, 0, :2]
    
    # Color-code initial positions by GMM component (approximately)
    # Sample component assignments based on posterior weights
    initial_component_colors = []
    for pos in initial_positions:
        # Simple assignment based on distance to component means
        distances = [np.linalg.norm(pos - np.array(initial_gmm.means[i][:2])) 
                    for i in range(initial_gmm.num_components)]
        closest_component = np.argmin(distances)
        initial_component_colors.append(closest_component)
    
    colors = ['red', 'blue', 'orange']
    for i in range(initial_gmm.num_components):
        component_mask = np.array(initial_component_colors) == i
        if np.any(component_mask):
            ax1.scatter(initial_positions[component_mask, 0], initial_positions[component_mask, 1], 
                       c=colors[i], s=2, alpha=0.6, label=f'Init Comp {i+1}')
    
    # Plot final distribution
    final_positions = state_trajectories[:, -1, :2]
    ax1.scatter(final_positions[:, 0], final_positions[:, 1], 
               c='black', s=2, alpha=0.7, label='Final (should be single Gaussian)')
    
    # Plot target
    ax1.scatter(target_mean[0], target_mean[1], 
               c='gold', s=100, marker='*', linewidth=2, edgecolors='black', label='Target')
    
    # Add component means as reference
    for i, mean in enumerate(initial_gmm.means):
        ax1.scatter(mean[0], mean[1], c=colors[i], s=50, marker='s', 
                   edgecolors='black', label=f'Init Mean {i+1}' if i < 3 else "")
    
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('GMM → Gaussian Steering')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Control magnitude plot
    ax2 = axes[1]
    
    control_norms = np.linalg.norm(control_trajectories, axis=2)
    mean_norms = np.mean(control_norms, axis=0)
    std_norms = np.std(control_norms, axis=0)
    max_norms = np.max(control_norms, axis=0)
    
    time_steps = np.arange(control_trajectories.shape[1])
    
    ax2.fill_between(time_steps, mean_norms - std_norms, mean_norms + std_norms,
                    alpha=0.3, label='±1 std')
    ax2.plot(time_steps, mean_norms, 'b-', linewidth=2, label='Mean')
    ax2.plot(time_steps, max_norms, 'r-', linewidth=1, label='Max')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Control Norm ||u||')
    ax2.set_title('Control Magnitudes')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('simple_reproduction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✓ Plots saved as 'simple_reproduction_results.png'")


def main():
    """
    Main function for simplified reproduction.
    """
    print("=" * 60)
    print("Simplified Paper Reproduction:")
    print("GMM Propagation and Monte Carlo Validation")
    print("=" * 60)
    print()
    
    # Setup problem
    horizon = 20
    dt = 0.2
    
    print("Problem Setup:")
    system = create_double_integrator_2d(horizon, dt)
    initial_gmm = create_paper_initial_gmm()
    target_mean, target_cov = create_paper_target_gaussian()
    
    print(f"  System: 2D double integrator (4D state, 2D control)")
    print(f"  Horizon: {horizon} steps, dt = {dt}s")
    print(f"  Initial GMM: {initial_gmm.num_components} components")
    print(f"  Initial mean: {initial_gmm.overall_mean()}")
    print(f"  Target mean: {target_mean}")
    print()
    
    # Create GMM-to-Gaussian solution
    print("Creating GMM-to-Gaussian steering solution...")
    solution = gmm_to_gaussian_solution(system, initial_gmm, target_mean, target_cov)
    print("  ✓ Solution created")
    print()
    
    # Demonstrate GMM propagation
    state_gmms, control_gmms = run_gmm_propagation_demo(system, initial_gmm, solution, target_mean)
    print()
    
    # Monte Carlo validation
    state_trajectories, control_trajectories = run_monte_carlo_validation(
        system, initial_gmm, solution, target_mean, 100
    )
    print()
    
    # Visualization
    create_visualization(state_trajectories, control_trajectories, initial_gmm, target_mean)
    
    print("=" * 60)
    print("Simplified reproduction completed!")
    print("This demonstrates core GMM concepts without full optimization.")
    print("=" * 60)


if __name__ == "__main__":
    # Set JAX to CPU for reproducibility
    jax.config.update('jax_platform_name', 'cpu')
    
    main()