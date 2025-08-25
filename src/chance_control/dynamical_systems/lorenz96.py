from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from diffrax import (AbstractSolver, AbstractStepSizeController,
                     ConstantStepSize, Tsit5)
from jaxtyping import Array, Float, Key, jaxtyped

from chance_control.dynamical_systems import (AbstractContinuousDynamicalSystem,
                                       AbstractDynamicalSystem)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def lorenz96_derivatives(
    x: Float[Array, "*batch dim"], F: float = 8.0
) -> Float[Array, "*batch dim"]:
    """Compute the derivatives for the Lorenz 96 model."""
    # Handle rolling for periodic boundary conditions
    x_plus_1 = jnp.roll(x, shift=-1, axis=-1)  # X_{i+1}
    x_minus_1 = jnp.roll(x, shift=1, axis=-1)  # X_{i-1}
    x_minus_2 = jnp.roll(x, shift=2, axis=-1)  # X_{i-2}

    # Compute derivatives according to Lorenz 96 formula
    derivatives = (x_plus_1 - x_minus_2) * x_minus_1 - x + F

    return derivatives


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def lorenz96_forward(
    x: Float[Array, "*batch dim"],
    F: float = 8.0,
    dt: float = 0.01,
    steps: int = 12,  # 12 steps of 0.01 = 0.12 time units
) -> Float[Array, "*batch dim"]:
    """Advance the Lorenz 96 system forward by integrating for specified time units using RK4."""

    def rk4_step(state, _):
        # RK4 integration for one step
        k1 = lorenz96_derivatives(state, F)
        k2 = lorenz96_derivatives(state + dt / 2 * k1, F)
        k3 = lorenz96_derivatives(state + dt / 2 * k2, F)
        k4 = lorenz96_derivatives(state + dt * k3, F)

        # Update state
        new_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_state, None

    # Perform multiple steps using scan
    final_state, _ = jax.lax.scan(rk4_step, x, None, length=steps)
    return final_state


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def lorenz96_generate(
    key: Key[Array, "..."],
    batch_size: int = 1000,
    dim: int = 40,
    F: float = 8.0,
    spin_up_steps: int = 100,
    dt: float = 0.01,
    steps: int = 12,
) -> Float[Array, "{batch_size} {dim}"]:
    initial_states = F + 0.01 * jax.random.normal(key, shape=(batch_size, dim))

    # Flow towards attractor
    def body_fn(i, val):
        return lorenz96_forward(val, F, dt, steps)

    # Use fori_loop for spin-up steps
    final_states = jax.lax.fori_loop(0, spin_up_steps, body_fn, initial_states)

    return final_states


@jaxtyped(typechecker=typechecker)
class Lorenz96(AbstractContinuousDynamicalSystem, strict=True):
    F: float = 8.0
    dt: float = 0.05
    # steps: int = 12
    dim: int = 40
    solver: AbstractSolver = Tsit5()
    stepsize_contoller: AbstractStepSizeController = ConstantStepSize()

    plot_limits: list = eqx.field(
        default_factory=lambda: [
            (-9.1022879437755, 15.627137075906784),
            (-10.5924605433048, 15.06699926099321),
            (-8.922207556187573, 15.107259147225637),
        ],
    )

    @property
    def dimension(self):
        return self.dim

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        state = jnp.full(self.dim, self.F).at[0].add(0.01)
        noise = (
            0
            if key is None
            else jax.random.multivariate_normal(
                key, mean=jnp.zeros(self.dimension), cov=jnp.eye(self.dimension)
            )
        )

        return state + noise

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def vector_field(
        self,
        t,
        x: Float[Array, "dim"],
        args: Any,
    ) -> Float[Array, "dim"]:
        """Compute the derivatives for the Lorenz 96 model."""

        x_plus_1 = jnp.roll(x, shift=-1, axis=-1)  # X_{i+1}
        x_minus_1 = jnp.roll(x, shift=1, axis=-1)  # X_{i-1}
        x_minus_2 = jnp.roll(x, shift=2, axis=-1)  # X_{i-2}

        derivatives = (x_plus_1 - x_minus_2) * x_minus_1 - x + self.F

        return derivatives
