"""
Affine feedback control policy with probabilistic selection.

This module implements the control policy from "Chance-Constrained Gaussian Mixture 
Steering to a Terminal Gaussian Distribution" (Kumagai & Oguri, 2024).

The policy is: u_{k,i}(x_0) = v_k + L_k^i * (x_0 - μ^g_0)
where the feedback gains L_k^i are selected probabilistically based on the initial state.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Shaped, jaxtyped

from .gmm_models import GaussianMixtureModel, posterior_weights


class AffineControlPolicy(eqx.Module, strict=True):
    """
    Affine control policy with probabilistic feedback gain selection.
    
    Policy: u_{k,i}(x_0) = v_k + L_k^i * (x_0 - μ^g_0)
    
    The feedback matrix L_k^i is chosen according to posterior weights:
    P[L_k = L_k^i | x_0] = λ_i(x_0)
    """
    
    feedforward_gains: Float[Array, "horizon control_dim"]  # V = [v_0, ..., v_{N-1}]
    feedback_gains: Float[Array, "num_components horizon control_dim state_dim"]  # L^i_k
    initial_gmm: GaussianMixtureModel
    
    def __init__(
        self,
        feedforward_gains: Float[Array, "horizon control_dim"],
        feedback_gains: Float[Array, "num_components horizon control_dim state_dim"],
        initial_gmm: GaussianMixtureModel,
    ):
        """
        Initialize the affine control policy.
        
        Args:
            feedforward_gains: Deterministic feedforward control sequence
            feedback_gains: Probabilistic feedback gains for each component and time step
            initial_gmm: Initial GMM distribution for computing posterior weights
        """
        self.feedforward_gains = feedforward_gains
        self.feedback_gains = feedback_gains
        self.initial_gmm = initial_gmm
        
        # Validate dimensions
        horizon, control_dim = feedforward_gains.shape
        num_components, horizon_fb, control_dim_fb, state_dim = feedback_gains.shape
        
        assert horizon == horizon_fb, "Horizon mismatch between feedforward and feedback"
        assert control_dim == control_dim_fb, "Control dimension mismatch"
        assert num_components == initial_gmm.num_components, "Component number mismatch"
        assert state_dim == initial_gmm.dimension, "State dimension mismatch"
    
    @property
    def horizon(self) -> int:
        """Time horizon."""
        return self.feedforward_gains.shape[0]
    
    @property
    def control_dim(self) -> int:
        """Control dimension."""
        return self.feedforward_gains.shape[1]
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.feedback_gains.shape[3]
    
    @property
    def num_components(self) -> int:
        """Number of GMM components."""
        return self.feedback_gains.shape[0]
    
    @jaxtyped(typechecker=typechecker)
    def compute_control(
        self,
        x0: Float[Array, "state_dim"],
        time_step: int,
        component_index: int | None = None,
    ) -> Float[Array, "control_dim"]:
        """
        Compute control input at given time step.
        
        Args:
            x0: Initial state observation
            time_step: Current time step (0 to horizon-1)
            component_index: If provided, use this component's gains directly
            
        Returns:
            Control input
        """
        mu_g = self.initial_gmm.overall_mean()
        v_k = self.feedforward_gains[time_step]
        
        if component_index is not None:
            # Use specified component
            L_k_i = self.feedback_gains[component_index, time_step]
            return v_k + L_k_i @ (x0 - mu_g)
        else:
            # Sample component based on posterior weights
            weights = posterior_weights(x0, self.initial_gmm)
            # For deterministic evaluation, use expected control
            expected_control = v_k
            for i in range(self.num_components):
                L_k_i = self.feedback_gains[i, time_step]
                expected_control += weights[i] * (L_k_i @ (x0 - mu_g))
            return expected_control
    
    @jaxtyped(typechecker=typechecker)
    def sample_component(
        self,
        key: Array,
        x0: Float[Array, "state_dim"],
    ) -> int:
        """
        Sample component index based on posterior weights.
        
        Args:
            key: Random key
            x0: Initial state observation
            
        Returns:
            Sampled component index
        """
        weights = posterior_weights(x0, self.initial_gmm)
        return jax.random.choice(key, self.num_components, p=weights)
    
    @jaxtyped(typechecker=typechecker)
    def control_sequence_deterministic(
        self,
        x0: Float[Array, "state_dim"],
        component_index: int,
    ) -> Float[Array, "horizon control_dim"]:
        """
        Generate deterministic control sequence for a specific component.
        
        Args:
            x0: Initial state
            component_index: Component to use for feedback gains
            
        Returns:
            Control sequence
        """
        mu_g = self.initial_gmm.overall_mean()
        deviation = x0 - mu_g
        
        controls = []
        for k in range(self.horizon):
            v_k = self.feedforward_gains[k]
            L_k_i = self.feedback_gains[component_index, k]
            u_k = v_k + L_k_i @ deviation
            controls.append(u_k)
        
        return jnp.stack(controls)
    
    @jaxtyped(typechecker=typechecker)
    def control_sequence_sampled(
        self,
        key: Array,
        x0: Float[Array, "state_dim"],
    ) -> Float[Array, "horizon control_dim"]:
        """
        Generate control sequence with probabilistically sampled component.
        
        Args:
            key: Random key
            x0: Initial state
            
        Returns:
            Control sequence
        """
        component_idx = self.sample_component(key, x0)
        return self.control_sequence_deterministic(x0, component_idx)
    
    @jaxtyped(typechecker=typechecker)
    def expected_control_sequence(
        self,
        x0: Float[Array, "state_dim"],
    ) -> Float[Array, "horizon control_dim"]:
        """
        Compute expected control sequence over all components.
        
        Args:
            x0: Initial state
            
        Returns:
            Expected control sequence
        """
        weights = posterior_weights(x0, self.initial_gmm)
        mu_g = self.initial_gmm.overall_mean()
        deviation = x0 - mu_g
        
        controls = []
        for k in range(self.horizon):
            v_k = self.feedforward_gains[k]
            expected_feedback = jnp.zeros_like(v_k)
            
            for i in range(self.num_components):
                L_k_i = self.feedback_gains[i, k]
                expected_feedback += weights[i] * (L_k_i @ deviation)
            
            u_k = v_k + expected_feedback
            controls.append(u_k)
        
        return jnp.stack(controls)


class ControlPolicyOptimizationVariables(eqx.Module, strict=True):
    """
    Structured representation of control policy optimization variables.
    
    These are the decision variables V and L^i for the convex optimization problem.
    """
    
    feedforward_gains: Float[Array, "horizon control_dim"]  # V
    feedback_gains: Float[Array, "num_components horizon control_dim state_dim"]  # L^i
    
    def __init__(
        self,
        horizon: int,
        control_dim: int,
        state_dim: int,
        num_components: int,
    ):
        """
        Initialize with zero variables.
        
        Args:
            horizon: Time horizon
            control_dim: Control dimension
            state_dim: State dimension  
            num_components: Number of GMM components
        """
        self.feedforward_gains = jnp.zeros((horizon, control_dim))
        self.feedback_gains = jnp.zeros((num_components, horizon, control_dim, state_dim))
    
    @jaxtyped(typechecker=typechecker)
    def to_policy(self, initial_gmm: GaussianMixtureModel) -> AffineControlPolicy:
        """
        Convert optimization variables to control policy.
        
        Args:
            initial_gmm: Initial GMM distribution
            
        Returns:
            Control policy
        """
        return AffineControlPolicy(
            self.feedforward_gains,
            self.feedback_gains,
            initial_gmm
        )
    
    @jaxtyped(typechecker=typechecker)
    def flatten(self) -> Float[Array, "total_vars"]:
        """
        Flatten variables for optimization.
        
        Returns:
            Flattened variable vector
        """
        v_flat = self.feedforward_gains.flatten()
        l_flat = self.feedback_gains.flatten()
        return jnp.concatenate([v_flat, l_flat])
    
    @classmethod
    @jaxtyped(typechecker=typechecker)
    def from_flat(
        cls,
        variables: Float[Array, "total_vars"],
        horizon: int,
        control_dim: int,
        state_dim: int,
        num_components: int,
    ) -> "ControlPolicyOptimizationVariables":
        """
        Reconstruct from flattened variables.
        
        Args:
            variables: Flattened variable vector
            horizon: Time horizon
            control_dim: Control dimension
            state_dim: State dimension
            num_components: Number of GMM components
            
        Returns:
            Reconstructed optimization variables
        """
        v_size = horizon * control_dim
        l_size = num_components * horizon * control_dim * state_dim
        
        assert variables.shape[0] == v_size + l_size, "Incorrect variable vector size"
        
        v_flat = variables[:v_size]
        l_flat = variables[v_size:]
        
        feedforward_gains = v_flat.reshape((horizon, control_dim))
        feedback_gains = l_flat.reshape((num_components, horizon, control_dim, state_dim))
        
        obj = cls.__new__(cls)
        obj.feedforward_gains = feedforward_gains
        obj.feedback_gains = feedback_gains
        
        return obj


@jaxtyped(typechecker=typechecker)
def simulate_monte_carlo_trajectories(
    policy: AffineControlPolicy,
    system,  # LinearDiscreteSystem
    initial_gmm: GaussianMixtureModel,
    key: Array,
    num_trajectories: int = 1000,
) -> tuple[Float[Array, "num_trajectories horizon+1 state_dim"], Float[Array, "num_trajectories horizon control_dim"]]:
    """
    Simulate Monte Carlo trajectories with the control policy.
    
    Args:
        policy: Control policy
        system: Discrete linear system
        initial_gmm: Initial GMM distribution
        key: Random key
        num_trajectories: Number of trajectories to simulate
        
    Returns:
        (state_trajectories, control_trajectories)
    """
    keys = jax.random.split(key, num_trajectories + 1)
    sample_key, control_keys = keys[0], keys[1:]
    
    # Sample initial states
    initial_states = initial_gmm.sample(sample_key, num_trajectories)
    
    # Generate control sequences
    def generate_trajectory(i):
        x0 = initial_states[i]
        control_seq = policy.control_sequence_sampled(control_keys[i], x0)
        
        # Simulate system trajectory
        times, states = system.trajectory(
            initial_time=0,
            final_time=policy.horizon,
            state=x0,
            control_sequence=control_seq
        )
        
        return states, control_seq
    
    # Vectorized trajectory generation
    all_states, all_controls = jax.vmap(generate_trajectory)(jnp.arange(num_trajectories))
    
    return all_states, all_controls