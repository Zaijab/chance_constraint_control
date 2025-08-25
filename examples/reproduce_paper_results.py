"""
Reproduction of numerical example from "Chance-Constrained Gaussian Mixture 
Steering to a Terminal Gaussian Distribution" (Kumagai & Oguri, 2024).

This script reproduces the 4D double integrator example with:
- Initial 3-component GMM distribution
- Terminal Gaussian distribution steering
- Affine state constraints and 2-norm control constraints
- Comparison of URA vs IRA solutions
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chance_control.linear_system import create_double_integrator_2d
from chance_control.gmm_models import create_paper_initial_gmm, create_paper_target_gaussian
from chance_control.chance_constraints import create_paper_constraints
from chance_control.optimization import create_paper_cost_matrices, ConvexOptimizationProblem
from chance_control.risk_allocation import IterativeRiskAllocation
from chance_control.control_policy import simulate_monte_carlo_trajectories


def setup_paper_problem():
    """
    Set up the problem parameters exactly as described in the paper.
    
    Returns:
        Tuple of (system, initial_gmm, target_mean, target_cov, constraints, costs)
    """
    # System parameters
    horizon = 20
    dt = 0.2
    
    # Create 2D double integrator system (4D state: [x, y, vx, vy])
    system = create_double_integrator_2d(horizon, dt)
    
    # Initial GMM distribution (3 components)
    initial_gmm = create_paper_initial_gmm()
    
    # Target terminal distribution
    target_mean, target_cov = create_paper_target_gaussian()
    
    # Constraints
    constraint_matrices, constraint_bounds, Delta, u_max, Gamma = create_paper_constraints()
    
    # Cost matrices (quadratic: Q=0, R=I)
    Q_matrices, R_matrices = create_paper_cost_matrices(horizon)
    
    print("Problem Setup:")
    print(f"  System: 2D double integrator (4D state, 2D control)")
    print(f"  Horizon: {horizon} steps, dt = {dt}s")
    print(f"  Initial GMM: {initial_gmm.num_components} components")
    print(f"  Target: μ_f = {target_mean}, Σ_f diagonal")
    print(f"  State constraints: {len(constraint_bounds)} affine, risk Δ = {Delta}")
    print(f"  Control constraints: ||u|| ≤ {u_max}, risk Γ = {Gamma}")
    print()
    
    return (
        system, initial_gmm, target_mean, target_cov,
        constraint_matrices, constraint_bounds, Delta, u_max, Gamma,
        Q_matrices, R_matrices
    )


def solve_without_constraints(system, initial_gmm, target_mean, target_cov, Q_matrices, R_matrices):
    """
    Solve the problem without chance constraints (unconstrained case).
    
    Returns:
        (success, solution, info)
    """
    print("Solving unconstrained problem...")
    
    opt_problem = ConvexOptimizationProblem(
        system, initial_gmm, target_mean, target_cov, system.horizon
    )
    
    success, solution, info = opt_problem.solve_problem(
        cost_type="quadratic",
        Q_matrices=Q_matrices,
        R_matrices=R_matrices,
        verbose=False
    )
    
    if success:
        print(f"  ✓ Solved successfully, cost = {info['objective_value']:.6f}")
    else:
        print(f"  ✗ Failed: {info.get('status', 'unknown error')}")
    
    return success, solution, info


def solve_with_ura(system, initial_gmm, target_mean, target_cov, 
                   constraint_matrices, constraint_bounds, Delta, u_max, Gamma,
                   Q_matrices, R_matrices):
    """
    Solve with uniform risk allocation (URA).
    
    Returns:
        (success, solution, info)
    """
    print("Solving with Uniform Risk Allocation (URA)...")
    
    opt_problem = ConvexOptimizationProblem(
        system, initial_gmm, target_mean, target_cov, system.horizon
    )
    
    success, solution, info = opt_problem.solve_problem(
        cost_type="quadratic",
        Q_matrices=Q_matrices,
        R_matrices=R_matrices,
        state_constraints=(constraint_matrices, constraint_bounds),
        control_bound=u_max,
        Delta=Delta,
        Gamma=Gamma,
        verbose=False
    )
    
    if success:
        print(f"  ✓ Solved successfully, cost = {info['objective_value']:.6f}")
    else:
        print(f"  ✗ Failed: {info.get('status', 'unknown error')}")
    
    return success, solution, info


def solve_with_ira(system, initial_gmm, target_mean, target_cov,
                   constraint_matrices, constraint_bounds, Delta, u_max, Gamma,
                   Q_matrices, R_matrices):
    """
    Solve with Iterative Risk Allocation (IRA).
    
    Returns:
        (success, solution, info)
    """
    print("Solving with Iterative Risk Allocation (IRA)...")
    
    ira = IterativeRiskAllocation(
        system, initial_gmm, target_mean, target_cov, system.horizon,
        beta=0.7, tolerance=1e-2, max_iterations=13
    )
    
    success, solution, info = ira.run_ira(
        constraint_matrices, constraint_bounds, u_max,
        total_affine_risk=Delta, total_control_risk=Gamma,
        cost_type="quadratic", Q_matrices=Q_matrices, R_matrices=R_matrices,
        verbose=True
    )
    
    if success:
        improvement = info.get('improvement_percent', 0)
        print(f"  ✓ Converged after {info['iterations']} iterations")
        print(f"  ✓ Final cost = {info['final_cost']:.6f}")
        print(f"  ✓ Improvement over URA: {improvement:.2f}%")
    else:
        print(f"  ✗ Failed to converge")
    
    return success, solution, info


def run_monte_carlo_simulation(policy, system, initial_gmm, num_trajectories=1000):
    """
    Run Monte Carlo simulation to validate the solution.
    
    Returns:
        (state_trajectories, control_trajectories)
    """
    print(f"Running Monte Carlo simulation ({num_trajectories} trajectories)...")
    
    key = jax.random.PRNGKey(42)
    
    state_trajs, control_trajs = simulate_monte_carlo_trajectories(
        policy, system, initial_gmm, key, num_trajectories
    )
    
    print(f"  ✓ Simulated {num_trajectories} trajectories")
    
    return state_trajs, control_trajs


def plot_results(state_trajs_unconstrained, state_trajs_ura, state_trajs_ira,
                control_trajs_unconstrained, control_trajs_ura, control_trajs_ira,
                constraint_matrices, constraint_bounds, target_mean, u_max):
    """
    Create plots comparing the different solutions.
    """
    print("Creating visualization plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # State trajectory plots
    for i, (state_trajs, title) in enumerate([
        (state_trajs_unconstrained, "Unconstrained"),
        (state_trajs_ura, "URA"),
        (state_trajs_ira, "IRA")
    ]):
        ax = axes[0, i]
        
        # Plot sample trajectories (position only)
        sample_indices = np.random.choice(state_trajs.shape[0], 100, replace=False)
        for idx in sample_indices:
            traj = state_trajs[idx, :, :2]  # x, y positions
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.1, linewidth=0.5)
        
        # Plot initial and final distributions
        initial_positions = state_trajs[:, 0, :2]
        final_positions = state_trajs[:, -1, :2]
        
        ax.scatter(initial_positions[:, 0], initial_positions[:, 1], 
                  c='green', s=1, alpha=0.5, label='Initial')
        ax.scatter(final_positions[:, 0], final_positions[:, 1], 
                  c='red', s=1, alpha=0.5, label='Final')
        ax.scatter(target_mean[0], target_mean[1], 
                  c='red', s=100, marker='x', linewidth=3, label='Target')
        
        # Plot state constraints
        if constraint_matrices is not None:
            x_range = ax.get_xlim()
            y_range = ax.get_ylim()
            
            for j, (a, b) in enumerate(zip(constraint_matrices, constraint_bounds)):
                if abs(a[1]) > 1e-6:  # y coefficient non-zero
                    x_line = np.linspace(x_range[0], x_range[1], 100)
                    y_line = (b - a[0] * x_line) / a[1]
                    ax.plot(x_line, y_line, 'r--', alpha=0.7, 
                           label=f'Constraint {j+1}' if i == 0 else "")
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'{title} - State Trajectories')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    # Control trajectory plots  
    for i, (control_trajs, title) in enumerate([
        (control_trajs_unconstrained, "Unconstrained"),
        (control_trajs_ura, "URA"),
        (control_trajs_ira, "IRA")
    ]):
        ax = axes[1, i]
        
        # Plot control norms over time
        control_norms = np.linalg.norm(control_trajs, axis=2)  # (num_traj, horizon)
        
        # Statistics
        mean_norms = np.mean(control_norms, axis=0)
        std_norms = np.std(control_norms, axis=0)
        max_norms = np.max(control_norms, axis=0)
        
        time_steps = np.arange(control_trajs.shape[1])
        
        ax.fill_between(time_steps, mean_norms - std_norms, mean_norms + std_norms,
                       alpha=0.3, label='±1 std')
        ax.plot(time_steps, mean_norms, 'b-', linewidth=2, label='Mean')
        ax.plot(time_steps, max_norms, 'r-', linewidth=1, label='Max')
        
        # Control bound
        ax.axhline(y=u_max, color='red', linestyle='--', alpha=0.7, 
                  label=f'Bound ({u_max})')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Control Norm ||u||')
        ax.set_title(f'{title} - Control Magnitudes')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('paper_reproduction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✓ Plots saved as 'paper_reproduction_results.png'")


def main():
    """
    Main function to reproduce the paper's numerical example.
    """
    print("=" * 60)
    print("Reproducing Paper Results:")
    print("Chance-Constrained Gaussian Mixture Steering")
    print("=" * 60)
    print()
    
    # Setup problem
    (system, initial_gmm, target_mean, target_cov,
     constraint_matrices, constraint_bounds, Delta, u_max, Gamma,
     Q_matrices, R_matrices) = setup_paper_problem()
    
    # Solve different variants
    results = {}
    
    # 1. Unconstrained
    success, solution, info = solve_without_constraints(
        system, initial_gmm, target_mean, target_cov, Q_matrices, R_matrices
    )
    if success:
        results['unconstrained'] = (solution, info)
        policy_unconstrained = solution.to_policy(initial_gmm)
    else:
        print("Skipping unconstrained case due to solve failure")
        policy_unconstrained = None
    
    print()
    
    # 2. URA  
    success, solution, info = solve_with_ura(
        system, initial_gmm, target_mean, target_cov,
        constraint_matrices, constraint_bounds, Delta, u_max, Gamma,
        Q_matrices, R_matrices
    )
    if success:
        results['ura'] = (solution, info)
        policy_ura = solution.to_policy(initial_gmm)
    else:
        print("Skipping URA case due to solve failure")
        policy_ura = None
    
    print()
    
    # 3. IRA
    success, solution, info = solve_with_ira(
        system, initial_gmm, target_mean, target_cov,
        constraint_matrices, constraint_bounds, Delta, u_max, Gamma,
        Q_matrices, R_matrices
    )
    if success:
        results['ira'] = (solution, info)
        policy_ira = solution.to_policy(initial_gmm)
    else:
        print("Skipping IRA case due to solve failure")
        policy_ira = None
    
    print()
    
    # Monte Carlo simulations
    trajectories = {}
    
    if policy_unconstrained is not None:
        state_trajs, control_trajs = run_monte_carlo_simulation(
            policy_unconstrained, system, initial_gmm, 1000
        )
        trajectories['unconstrained'] = (state_trajs, control_trajs)
    
    if policy_ura is not None:
        state_trajs, control_trajs = run_monte_carlo_simulation(
            policy_ura, system, initial_gmm, 1000
        )
        trajectories['ura'] = (state_trajs, control_trajs)
    
    if policy_ira is not None:
        state_trajs, control_trajs = run_monte_carlo_simulation(
            policy_ira, system, initial_gmm, 1000  
        )
        trajectories['ira'] = (state_trajs, control_trajs)
    
    print()
    
    # Create visualizations
    if len(trajectories) >= 2:
        # Get trajectories for plotting
        state_unconstrained = trajectories.get('unconstrained', (None, None))[0]
        control_unconstrained = trajectories.get('unconstrained', (None, None))[1]
        state_ura = trajectories.get('ura', (None, None))[0]
        control_ura = trajectories.get('ura', (None, None))[1]
        state_ira = trajectories.get('ira', (None, None))[0]
        control_ira = trajectories.get('ira', (None, None))[1]
        
        # Use available results for comparison
        if state_ura is not None or state_ira is not None:
            plot_results(
                state_unconstrained, state_ura, state_ira,
                control_unconstrained, control_ura, control_ira,
                constraint_matrices, constraint_bounds, target_mean, u_max
            )
    
    # Print summary
    print("=" * 60)
    print("RESULTS SUMMARY:")
    print("=" * 60)
    
    for method, (solution, info) in results.items():
        cost = info.get('objective_value', info.get('final_cost', 'N/A'))
        print(f"{method.upper():>15}: Cost = {cost}")
        
        if method == 'ira' and 'improvement_percent' in info:
            print(f"{'':>15}  Improvement over URA: {info['improvement_percent']:.2f}%")
    
    print()
    print("Paper reproduction completed!")
    
    return results


if __name__ == "__main__":
    # Set JAX to CPU for reproducibility
    jax.config.update('jax_platform_name', 'cpu')
    
    main()