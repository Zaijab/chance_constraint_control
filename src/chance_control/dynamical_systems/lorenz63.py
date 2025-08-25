import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from diffrax import (AbstractSolver, AbstractStepSizeController,
                     ConstantStepSize, ODETerm, SaveAt, Tsit5, diffeqsolve)
from jaxtyping import Array, Float, Key, jaxtyped

from chance_control.dynamical_systems import AbstractContinuousDynamicalSystem


@jaxtyped(typechecker=typechecker)
class Lorenz63(AbstractContinuousDynamicalSystem, strict=True):
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    solver: AbstractSolver = Tsit5()
    stepsize_contoller: AbstractStepSizeController = ConstantStepSize()

    plot_limits: list = eqx.field(
        default_factory=lambda: [
            (-20.594663510210776, 20.272689840576522),
            (-28.857151028514547, 26.829824666076732),
            (-0.019340876766049453, 48.682780695599895),
        ],
    )

    @property
    def dimension(self):
        return 3

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        state = jnp.array([8.0, 0.0, 0.0])

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
    def vector_field(self, t, y, args):
        x, y, z = y
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return jnp.array([dx, dy, dz])
