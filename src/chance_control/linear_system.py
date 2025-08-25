"""
Linear time-invariant discrete dynamical system for chance-constrained control.

Implements the discrete-time linear system:
    x_{k+1} = A_k * x_k + B_k * u_k

This system is used in the paper "Chance-Constrained Gaussian Mixture Steering 
to a Terminal Gaussian Distribution" (Kumagai & Oguri, 2024).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from diffrax import SaveAt
from jaxtyping import Array, Float, Shaped, jaxtyped

from .dynamical_systems.dynamical_system_abc import AbstractDynamicalSystem


class LinearDiscreteSystem(AbstractDynamicalSystem, strict=True):
    """
    Discrete-time linear time-varying system: x_{k+1} = A_k * x_k + B_k * u_k
    
    For time-invariant case, A_matrices and B_matrices contain repeated matrices.
    For time-varying case, they contain the sequence of matrices over the horizon.
    """
    
    A_matrices: Float[Array, "horizon state_dim state_dim"]
    B_matrices: Float[Array, "horizon state_dim control_dim"] 
    _state_dim: int
    _control_dim: int
    horizon: int
    
    def __init__(
        self,
        A_matrices: Float[Array, "horizon state_dim state_dim"],
        B_matrices: Float[Array, "horizon state_dim control_dim"],
    ):
        """
        Initialize the linear discrete system.
        
        Args:
            A_matrices: State transition matrices for each time step
            B_matrices: Control input matrices for each time step
        """
        self.A_matrices = A_matrices
        self.B_matrices = B_matrices
        self.horizon = A_matrices.shape[0]
        self._state_dim = A_matrices.shape[1]
        self._control_dim = B_matrices.shape[2]
        
        # Validate dimensions
        assert A_matrices.shape == (self.horizon, self._state_dim, self._state_dim)
        assert B_matrices.shape == (self.horizon, self._state_dim, self._control_dim)
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the state space."""
        return self._state_dim
    
    @property
    def control_dimension(self) -> int:
        """Return the dimension of the control space."""
        return self._control_dim
    
    @jaxtyped(typechecker=typechecker)
    def initial_state(
        self,
        key: Array | None = None,
        mean: Float[Array, "state_dim"] | None = None,
        cov: Float[Array, "state_dim state_dim"] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        """
        Generate an initial state.
        
        Args:
            key: Random key for sampling (if None, returns mean or zero)
            mean: Mean of initial distribution
            cov: Covariance of initial distribution
            
        Returns:
            Initial state vector
        """
        if mean is None:
            mean = jnp.zeros(self._state_dim)
            
        if key is None:
            return mean
            
        if cov is None:
            cov = 0.1 * jnp.eye(self._state_dim)
            
        return jax.random.multivariate_normal(key, mean, cov)
    
    def step(
        self,
        state: Float[Array, "state_dim"],
        control: Float[Array, "control_dim"],
        time_step: int,
    ) -> Float[Array, "state_dim"]:
        """
        Single time step of the discrete system.
        
        Args:
            state: Current state
            control: Control input
            time_step: Current time step (0 to horizon-1)
            
        Returns:
            Next state
        """
        A_k = self.A_matrices[time_step]
        B_k = self.B_matrices[time_step]
        return A_k @ state + B_k @ control
    
    @jaxtyped(typechecker=typechecker)
    def trajectory(
        self,
        initial_time: float | Shaped[Array, ""] | int,
        final_time: float | Shaped[Array, ""] | int,
        state: Float[Array, "state_dim"],
        control_sequence: Float[Array, "horizon control_dim"],
        saveat: SaveAt | None = None,
    ) -> tuple[Float[Array, "..."], Float[Array, "... state_dim"]]:
        """
        Compute trajectory given initial state and control sequence.
        
        Args:
            initial_time: Starting time (typically 0)
            final_time: Final time (typically horizon)
            state: Initial state
            control_sequence: Sequence of control inputs
            saveat: Save specification (if None, saves all steps)
            
        Returns:
            (times, states) tuple
        """
        initial_time = int(initial_time) if isinstance(initial_time, (float, int)) else int(initial_time.item())
        final_time = int(final_time) if isinstance(final_time, (float, int)) else int(final_time.item())
        
        num_steps = final_time - initial_time
        times = jnp.arange(initial_time, final_time + 1, dtype=jnp.float32)
        
        def scan_fn(carry_state, inputs):
            time_step, control = inputs
            next_state = self.step(carry_state, control, time_step)
            return next_state, carry_state
        
        time_indices = jnp.arange(initial_time, final_time)
        controls = control_sequence[initial_time:final_time]
        inputs = (time_indices, controls)
        
        final_state, states = jax.lax.scan(scan_fn, state, inputs)
        
        # Include final state
        all_states = jnp.concatenate([states, final_state[None, :]], axis=0)
        
        return times, all_states
    
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def concatenated_matrices(self) -> tuple[
        Float[Array, "total_state_dim state_dim"],
        Float[Array, "total_state_dim total_control_dim"]
    ]:
        """
        Create concatenated formulation matrices A and B for vectorized computation.
        
        Following the paper's concatenated formulation:
        X = [x_0, x_1, ..., x_N]^T = A * x_0 + B * U
        where U = [u_0, u_1, ..., u_{N-1}]^T
        
        Returns:
            (A, B) matrices for concatenated formulation
        """
        N = self.horizon
        nx = self._state_dim
        nu = self._control_dim
        
        # A matrix: (N+1)*nx x nx
        A = jnp.zeros(((N + 1) * nx, nx))
        
        # B matrix: (N+1)*nx x N*nu  
        B = jnp.zeros(((N + 1) * nx, N * nu))
        
        # First row: x_0 = I * x_0
        A = A.at[:nx, :].set(jnp.eye(nx))
        
        # Build A matrix recursively
        A_prod = jnp.eye(nx)
        for k in range(N):
            A_k = self.A_matrices[k]
            A_prod = A_k @ A_prod
            row_start = (k + 1) * nx
            row_end = (k + 2) * nx
            A = A.at[row_start:row_end, :].set(A_prod)
        
        # Build B matrix
        for k in range(N):
            # For state x_{k+1}, affected by controls u_0, ..., u_k
            row_start = (k + 1) * nx
            row_end = (k + 2) * nx
            
            for j in range(k + 1):
                col_start = j * nu
                col_end = (j + 1) * nu
                
                # Compute product A_{k} * A_{k-1} * ... * A_{j+1} * B_j
                B_contrib = self.B_matrices[j]
                for i in range(j + 1, k + 1):
                    B_contrib = self.A_matrices[i] @ B_contrib
                
                B = B.at[row_start:row_end, col_start:col_end].set(B_contrib)
        
        return A, B


@jaxtyped(typechecker=typechecker)
def create_double_integrator_2d(
    horizon: int,
    dt: float = 0.2,
) -> LinearDiscreteSystem:
    """
    Create a 2D double integrator system as used in the paper's numerical example.
    
    State: [x, y, vx, vy]^T
    Control: [ax, ay]^T
    
    Dynamics:
    x_{k+1} = x_k + vx_k * dt + 0.5 * ax_k * dt^2
    y_{k+1} = y_k + vy_k * dt + 0.5 * ay_k * dt^2
    vx_{k+1} = vx_k + ax_k * dt
    vy_{k+1} = vy_k + ay_k * dt
    
    Args:
        horizon: Time horizon
        dt: Time step
        
    Returns:
        LinearDiscreteSystem for 2D double integrator
    """
    # State transition matrix
    A = jnp.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Control input matrix
    B = jnp.array([
        [dt**2 / 2, 0.0],
        [0.0, dt**2 / 2],
        [dt, 0.0],
        [0.0, dt]
    ])
    
    # Repeat for all time steps (time-invariant)
    A_matrices = jnp.tile(A[None, :, :], (horizon, 1, 1))
    B_matrices = jnp.tile(B[None, :, :], (horizon, 1, 1))
    
    return LinearDiscreteSystem(A_matrices, B_matrices)